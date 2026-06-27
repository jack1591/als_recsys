"""

"""

from pathlib import Path

from utils import load_and_prepare_data


def main():

    print("\n" + "=" * 70)
    print("INSTACART DATA PREPARATION FOR TUTORIALS")
    print("=" * 70)

    raw_data_path = 'data/raw'
    processed_data_path = 'data/processed'

    required_files = [
        'orders.csv',
        'order_products__train.csv',
        'products.csv',
        'aisles.csv',
        'departments.csv'
    ]

    print("Processing steps:")
    print("  1. Loading raw data")
    print("  2. Filtering users (5-20 orders)")
    print("  3. Selecting top 800 products")
    print("  4. Building interaction matrix")
    print("  5. Creating train/test split")
    print("  6. Generating user features")
    print("  7. Saving processed files")

    try:
        # Call the preparation function
        stats = load_and_prepare_data(
            raw_data_path=raw_data_path,
            processed_data_path=processed_data_path,
            n_users=80000,  # previously 20000 (also try 40k or 80k)
            n_products=1500,  # previously 800 (also try 1200 or even 2000)
            min_orders=5,
            max_orders=20
        )

        print("\nDataset Statistics:")
        print(f"  Users: {stats['n_users']:,}")
        print(f"  Products: {stats['n_products']:,}")
        print(f"  Interactions: {stats['n_interactions']:,}")
        print(f"  Sparsity: {stats['sparsity']:.2%}")

        print(f"\nProcessed data saved to: {processed_data_path}/")

    except Exception as e:

        print(f"\nError: {str(e)}")

        import traceback
        print("\nFull error trace:")
        traceback.print_exc()


if __name__ == "__main__":
    """
    Execute data preparation.
    """
    main()
