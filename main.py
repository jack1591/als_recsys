from utils import *
from model_ALS import *

def main():

    # STEP 1: LOAD DATA

    processed_path = 'data/processed'

    data = load_processed_data(processed_path)

    train_matrix = data['train_matrix']
    test_matrix = data['test_matrix']
    mappings = data['mappings']
    product_info = data['product_info']

    # STEP 2: TRAIN ALS MODEL

    als_model = train_als(
        train_matrix=train_matrix,
        factors=64,
        regularization=0.01,
        iterations=20,
        use_gpu=False
    )

    with open('model/als_model.pkl', 'wb') as f:
        pickle.dump(als_model, f)

    # STEP 3: GENERATE RECOMMENDATIONS

    test_user_ids = list(range(train_matrix.shape[0]))

    predictions, scores = predict_als(
        model=als_model,
        train_matrix=train_matrix,
        user_ids=test_user_ids,
        k=10,
        filter_already_purchased=True
    )

    # STEP 4: EVALUATE MODEL

    results = evaluate_model(
        predictions_dict=predictions,
        test_matrix=test_matrix,
        k_values=[5,10]
    )

    print_evaluation_summary(results, model_name="ALS")


if __name__ == '__main__':
    main()