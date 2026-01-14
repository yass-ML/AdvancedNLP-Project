import pandas as pd
import os
import glob
import numpy as np
from datasets import load_dataset

def create_splits():
    # Define paths
    base_dir = "datasets/competition_math/data"
    os.makedirs(base_dir, exist_ok=True)

    # Paths for output
    train_path = os.path.join(base_dir, "train.parquet")
    test_path = os.path.join(base_dir, "test.parquet")

    # Check if they already exist
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Splits already exist at {base_dir}. Skipping generation.")
        return

    print("Checking for local parquet files...")
    search_path = os.path.join(base_dir, "*.parquet")
    files = glob.glob(search_path)
    original_files = [f for f in files if "train.parquet" not in f and "test.parquet" not in f]

    df = None

    if original_files:
        print(f"Found local files: {original_files}. Loading...")
        dfs = [pd.read_parquet(f) for f in original_files]
        df = pd.concat(dfs, ignore_index=True)
    else:
        print("No local files found. Downloading 'hendrycks/competition_math' from Hugging Face...")
        try:
            # Download both splits
            print("Downloading 'qwedsacf/competition_math'...")
            dataset = load_dataset("qwedsacf/competition_math")

            print(f"Available splits: {list(dataset.keys())}")

            dfs_to_concat = []
            if 'train' in dataset:
                dfs_to_concat.append(dataset['train'].to_pandas())
            if 'test' in dataset:
                dfs_to_concat.append(dataset['test'].to_pandas())

            if not dfs_to_concat:
                 raise ValueError("Dataset has no 'train' or 'test' splits.")

            df = pd.concat(dfs_to_concat, ignore_index=True)

        except Exception as e:
            print(f"Failed to download dataset: {e}")
            return

    if df is None or len(df) == 0:
        print("Error: DataFrame is empty.")
        return

    print(f"Total examples available: {len(df)}")

    # Ensure required columns exist
    # HF dataset has 'problem', 'level', 'type', 'solution'
    if 'type' not in df.columns:
        print("Warning: 'type' column missing. Checking for 'category'...")
        if 'category' in df.columns:
            df.rename(columns={'category': 'type'}, inplace=True)

    # Simple random split (10% test)
    validation_split = 0.1

    # Shuffle indices
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    split_point = int(len(df) * (1 - validation_split))

    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    print(f"Generated Train set size: {len(train_df)}")
    print(f"Generated Test set size: {len(test_df)}")

    # Save
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    print(f"Successfully saved splits to:\n  - {train_path}\n  - {test_path}")

if __name__ == "__main__":
    create_splits()
