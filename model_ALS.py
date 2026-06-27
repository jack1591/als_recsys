"""
Training ALS model for collaborative filtering using the implicit

"""

import numpy as np
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight

def train_als(train_matrix, factors = 64, regularization = 0.01,
              iterations = 25, use_gpu = False):

    # уберем смещение у популярныхз товаров (чтобы модель не рекомендовала их слишком часто)
    train_matrix_weighted = bm25_weight(train_matrix, K1 = 50, B = 0.8)
    train_matrix_csr = train_matrix_weighted.tocsr()

    model = AlternatingLeastSquares(
        factors = factors,
        regularization = regularization,
        iterations = iterations,
        use_gpu = use_gpu,
        random_state = 42
    )

    model.fit(train_matrix_csr, show_progress=True)

    return model

def predict_als(model, train_matrix, user_ids = None, k = 10,
                filter_already_purchased = True):
    """
    Generate top-k recommendations for user with ALS model.

    :param model: Trained ALS model
    :param train_matrix: Training data to filter already purchased items
    :param user_ids: Ids of users to predict items fot
    :param k: Number of recommendations per user
    :param filter_already_purchased: If true, exclude items the user already purchased
    :return:
        predictions: Products ate ordered by predicting score
        scores: Confidence for each recommendation

    """

    if user_ids is None:
        user_ids = list(range(train_matrix.shape[0]))

    predictions = {}
    scores = {}

    train_matrix_csr = train_matrix.tocsr()

    print(f'Predicting for {len(user_ids)} users')

    batch_items, batch_scores = model.recommend(
        userid = user_ids,
        user_items = train_matrix_csr,
        N = k,
        filter_already_liked_items = filter_already_purchased
    )

    for i, user_idx in enumerate(user_ids):
        predictions[user_idx] = batch_items[i].tolist()
        scores[user_idx] = batch_scores[i].tolist()

    return predictions, scores