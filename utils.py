from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import pickle


def load_and_prepare_data(raw_data_path, processed_data_path,
                          n_users = 20000, n_products = 800,
                          min_orders = 5, max_orders = 25):
    Path(processed_data_path).mkdir(parents = True, exist_ok = True)

    orders = pd.read_csv(f"{raw_data_path}/orders.csv")
    order_products = pd.read_csv(f'{raw_data_path}/order_products__train.csv')
    products = pd.read_csv(f'{raw_data_path}/products.csv')
    aisles = pd.read_csv(f'{raw_data_path}/aisles.csv')
    departments = pd.read_csv(f'{raw_data_path}/departments.csv')

    # Merge orders with products first
    print(f"\nMerging orders with products...")

    order_data = orders.merge(order_products, on = 'order_id')

    print(f"  Total interactions: {len(order_data):,}")

    user_products_count = order_data.groupby('user_id')['product_id'].count()

    valid_users = user_products_count[
        (user_products_count >= min_orders) &
        (user_products_count <= max_orders)
    ].index

    selected_users = np.random.choice(valid_users, n_users, replace = False)

    order_data = order_data[order_data['user_id'].isin(selected_users)]

    # Now select top products from this filtered set
    products_count = order_data['product_id'].value_counts()
    top_products =products_count.head(n_products).index

    order_data = order_data[order_data['product_id'].isin(top_products)]

    print(f"\n  Final dataset:")
    print(f"    Products: {len(order_data['product_id'].unique())}")
    print(f"    Users: {len(order_data['user_id'].unique()):,}")
    print(f"    Interactions: {len(order_data):,}")

    filtered_orders = orders[orders['user_id'].isin(selected_users)]

    # Create user and item mappings
    unique_users = sorted(order_data['user_id'].unique())
    unique_products = sorted(order_data['product_id'].unique())

    n_users_final = len(unique_users)
    n_products_final = len(unique_products)

    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    idx_to_user = {idx: user_id for idx, user_id in enumerate(unique_users)}

    product_to_idx = {product_id: idx for idx, product_id in enumerate(unique_products)}
    idx_to_product = {idx: product_id for idx, product_id in enumerate(unique_products)}

    order_data['user_idx'] = order_data['user_id'].map(user_to_idx)
    order_data['product_idx'] = order_data['product_id'].map(product_to_idx)

    print("\nBuilding user-item interaction matrix...")
    interactions_count = order_data.groupby(['user_idx', 'product_idx']).size()

    rows = interactions_count.index.get_level_values(0)
    cols = interactions_count.index.get_level_values(1)
    data = interactions_count.values

    user_item_matrix = sp.csr_matrix(
        (data, (rows, cols)),
        shape=(n_users_final, n_products_final)  # ← фактические размеры
    )

    sparsity = 1 - (user_item_matrix.nnz / (n_users * n_products))
    print(f"  Matrix shape: {user_item_matrix.shape}")
    print(f"  Non-zero entries: {user_item_matrix.nnz:,}")
    print(f"  Sparsity: {sparsity:.2%}")

    # Create train/test split
    train_matrix, test_matrix = create_train_test_split(
        order_data, user_to_idx, product_to_idx,
        n_users_final, n_products_final  # ← передаём фактические
    )

    # Save matrices
    sp.save_npz(f'{processed_data_path}/user_item_matrix.npz', user_item_matrix)
    sp.save_npz(f'{processed_data_path}/train_matrix.npz', train_matrix)
    sp.save_npz(f'{processed_data_path}/test_matrix.npz', test_matrix)

    # Save mappings
    with open(f'{processed_data_path}/user_mapping.pkl', 'wb') as f:
        pickle.dump({'user_to_idx': user_to_idx, 'idx_to_user': idx_to_user}, f)

    with open(f'{processed_data_path}/product_mapping.pkl', 'wb') as f:
        pickle.dump({'product_to_idx': product_to_idx, 'idx_to_product': idx_to_product}, f)

    # Prepare and save product metadata
    print("\nPreparing product metadata...")
    product_info = products[products['product_id'].isin(top_products)].copy()
    product_info = product_info.merge(aisles, on='aisle_id')
    product_info = product_info.merge(departments, on='department_id')
    product_info['product_idx'] = product_info['product_id'].map(product_to_idx)
    product_info.to_csv(f"{processed_data_path}/product_info.csv", index=False)

    print("DATA PREPARATION COMPLETE")

    return {
        'n_users': n_users,
        'n_products': n_products,
        'n_interactions':user_item_matrix.nnz,
        'sparsity': sparsity
    }

def load_processed_data(processed_data_path):
    """
    Load processed data from disk

    :param processed_data_path: Path to folder containing processed data
    :return:
        dict containing:
        - train_matrix : scipy sparse matrix (users × products)
        - test_matrix : scipy sparse matrix (users × products)
        - mappings : dict with user and product mappings
        - product_info : DataFrame with product metadata
        - user_features : DataFrame with user features
    """

    train_matrix = sp.load_npz(f'{processed_data_path}/train_matrix.npz')
    test_matrix = sp.load_npz(f'{processed_data_path}/test_matrix.npz')

    with open(f'{processed_data_path}/user_mapping.pkl', 'rb') as f:
        user_mapping = pickle.load(f)

    with open(f'{processed_data_path}/product_mapping.pkl', 'rb') as f:
        product_mapping = pickle.load(f)

    product_info = pd.read_csv(f'{processed_data_path}/product_info.csv')

    return {
        'train_matrix': train_matrix,
        'test_matrix': test_matrix,
        'mappings': {
            'user_to_idx': user_mapping['user_to_idx'],
            'idx_to_user': user_mapping['idx_to_user'],
            'product_to_idx': product_mapping['product_to_idx'],
            'idx_to_product': product_mapping['idx_to_product'],
        },
        'product_info': product_info
    }

def create_train_test_split(order_data, user_to_idx, product_to_idx,
                            n_users, n_products, test_ratio = 0.2):
    train_rows, train_cols, train_data = [], [], []
    test_rows, test_cols, test_data = [], [], []

    for user_id, user_data in order_data.groupby('user_id'):
        user_idx = user_to_idx.get(user_id)

        user_products = user_data['product_id'].values
        n_user_products = len(user_products)

        if n_user_products < 2:
            # If only 1 product, put it in training
            product_idx = product_to_idx.get(user_products[0])
            if product_idx is not None:
                train_rows.append(user_idx)
                train_cols.append(product_idx)
                train_data.append(1)
            continue

        shuffled_products = np.random.permutation(user_products)
        n_test = max(1, int(n_user_products * test_ratio))

        test_products = shuffled_products[:n_test]
        train_products = shuffled_products[n_test:]

        for product_id in train_products:
            product_idx = product_to_idx.get(product_id)

            if product_idx is not None:
                train_rows.append(user_idx)
                train_cols.append(product_idx)
                train_data.append(1)

        for product_id in test_products:
            product_idx = product_to_idx.get(product_id)

            if product_idx is not None:
                test_rows.append(user_idx)
                test_cols.append(product_idx)
                test_data.append(1)

    train_matrix = sp.csr_matrix(
        (train_data, (train_rows, train_cols)),
        shape = (n_users, n_products)
    )

    test_matrix = sp.csr_matrix(
        (test_data, (test_rows, test_cols)),
        shape = (n_users, n_products)
    )

    train_matrix = (train_matrix > 0).astype(int)
    test_matrix = (test_matrix > 0).astype(int)

    print(f"  Train: {train_matrix.nnz:,} interactions")
    print(f"  Test: {test_matrix.nnz:,} interactions")

    return train_matrix, test_matrix


def calculate_ndcg(predictions_dict, ground_truth_matrix, k_values=[5, 10]):
    """
    Calculate NDCG@K for multiple K values.

    This is the core evaluation metric connecting to Tutorial 1.
    NDCG measures how well the model ranks relevant items at the top.

    Parameters:
    -----------
    predictions_dict : dict
        {user_idx: [list of product_idx in ranked order]}
    ground_truth_matrix : scipy sparse matrix
        Binary matrix where 1 = user purchased product
    k_values : list
        K values to evaluate (e.g., [5, 10])

    Returns:
    --------
    dict : {f'ndcg@{k}': score} for each k
    """

    ndcg_scores = {f'ndcg@{k}': [] for k in k_values}

    n_products = ground_truth_matrix.shape[1]

    # OPTIMIZATION: Only process users who have predictions
    for user_idx, pred_products in predictions_dict.items():
        # Ground truth: which products did this user buy?
        true_products = ground_truth_matrix[user_idx].toarray().flatten()

        # If user has no test purchases, skip
        if true_products.sum() == 0:
            continue

        # Create score array: 1 for predicted items in rank order, 0 otherwise
        pred_scores = np.zeros(n_products)
        for rank, product_idx in enumerate(pred_products):
            # Higher scores for items ranked earlier
            pred_scores[product_idx] = len(pred_products) - rank

        # Calculate NDCG for each K
        for k in k_values:
            try:
                ndcg = ndcg_score([true_products], [pred_scores], k=k)
                ndcg_scores[f'ndcg@{k}'].append(ndcg)
            except:
                # Handle edge cases
                pass

    # Average across all users
    results = {}
    for k in k_values:
        key = f'ndcg@{k}'
        if len(ndcg_scores[key]) > 0:
            results[key] = np.mean(ndcg_scores[key])
        else:
            results[key] = 0.0

    return results

def evaluate_model(predictions_dict, test_matrix, k_values = [5, 10]):
    """
    Model evaluation on NGCG and Coverage

    :param predictions_dict: {user_idx: [list of product_idx]}
    :param test_matrix: Ground truth test set
    :param k_values: k values to evaluate
    :return:
        All evaluation metrics
    """

    results = {}

    ndcg_results = calculate_ndcg(predictions_dict, test_matrix, k_values)
    results.update(ndcg_results)

    for k in k_values:
        precision_scores = []
        recall_scores = []

        for user_idx, pred_items in predictions_dict.items():
            # Limit to top K
            pred_k = pred_items[:k]

            # Ground truth items
            true_items = test_matrix[user_idx].nonzero()[1]

            if len(true_items) == 0:
                continue

            # Calculate hits
            hits = len(set(pred_k) & set(true_items))

            precision = hits / k if k > 0 else 0
            recall = hits / len(true_items) if len(true_items) > 0 else 0

            precision_scores.append(precision)
            recall_scores.append(recall)

        results[f'precision@{k}'] = np.mean(precision_scores) if precision_scores else 0
        results[f'recall@{k}'] = np.mean(recall_scores) if recall_scores else 0

    # Coverage: what fraction of items are ever recommended?
    all_recommended_items = set()
    for pred_items in predictions_dict.values():
        all_recommended_items.update(pred_items)

    results['coverage'] = len(all_recommended_items) / test_matrix.shape[1]

    return results


def print_evaluation_summary(results, model_name="Model"):
    """
    Print formatted evaluation results.

    Parameters:
    -----------
    results : dict
        Evaluation metrics
    model_name : str
        Name of the model for display
    """

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} EVALUATION RESULTS")
    print("=" * 70)

    # NDCG scores
    print("\nNDCG Scores (Primary Metric):")
    for k in [5, 10]:
        key = f'ndcg@{k}'
        if key in results:
            print(f"  NDCG@{k:2d} = {results[key]:.4f}")

    # Precision and Recall
    print("\nPrecision & Recall:")
    for k in [5, 10]:
        prec_key = f'precision@{k}'
        rec_key = f'recall@{k}'
        if prec_key in results and rec_key in results:
            print(f"  @{k:2d}: Precision = {results[prec_key]:.4f}, Recall = {results[rec_key]:.4f}")

    # Coverage
    if 'coverage' in results:
        print(f"\nCatalog Coverage: {results['coverage']:.2%}")

    print("=" * 70)

def recommend_for_user(model, user_id, train_matrix, k=10):
    """Top-k рекомендаций: возвращает (item_ids, scores)."""
    train_csr = train_matrix.tocsr()
    return model.recommend(
        userid=user_id,
        user_items=train_csr[user_id],
        N=k,
        filter_already_liked_items=True
    )


# ============================================================
# ГРАФИКИ
# ============================================================

def plot_ndcg_comparison(results_dict):
    """Столбцы NDCG. results_dict = {'ALS': 0.25, 'Random': 0.08}"""
    names, values = zip(*results_dict.items())
    plt.figure(figsize=(6, 4))
    bars = plt.bar(names, values, color='steelblue')
    for b, v in zip(bars, values):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.003,
                 f'{v:.4f}', ha='center', fontsize=10)
    plt.ylabel('NDCG@k')
    plt.title('Сравнение моделей')
    plt.tight_layout()
    plt.show()


def plot_embedding_pca(model, labels=None):
    """PCA item-эмбеддингов. labels — dict {idx: category} для раскраски."""
    from sklearn.decomposition import PCA
    coords = PCA(2).fit_transform(model.item_factors)

    plt.figure(figsize=(8, 6))
    if labels:
        cats = [labels.get(i, '?') for i in range(len(coords))]
        for cat in set(cats):
            mask = [c == cat for c in cats]
            plt.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.6, label=cat)
        plt.legend(fontsize=7, bbox_to_anchor=(1.05, 1))
    else:
        plt.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.5)
    plt.title('PCA item embeddings')
    plt.tight_layout()
    plt.show()


def plot_popular_items(train_matrix, top_k=20):
    """Горизонтальный барчарт популярных товаров."""
    pop = np.asarray(train_matrix.sum(axis=0)).ravel()
    top = np.argsort(pop)[::-1][:top_k]

    plt.figure(figsize=(8, 5))
    plt.barh(range(len(top)), pop[top], color='steelblue')
    plt.yticks(range(len(top)), [f'item_{i}' for i in top], fontsize=8)
    plt.gca().invert_yaxis()
    plt.xlabel('Число покупателей')
    plt.title(f'Топ-{top_k} товаров')
    plt.tight_layout()
    plt.show()