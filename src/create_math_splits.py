import pandas as pd
import os
import glob
import numpy as np

def create_splits():
    # Define paths
    base_dir = "datasets/competition_math/data"
    # Find the original parquet file
    search_path = os.path.join(base_dir, "*.parquet")
    files = glob.glob(search_path)

    # Filter out already created train/test files to avoid re-splitting splits
    original_files = [f for f in files if "train.parquet" not in f and "test.parquet" not in f]

    if not original_files:
        print("No original dataset file found or splits already exist (and original removed?).")
        # Check if train/test exist
        if os.path.exists(os.path.join(base_dir, "train.parquet")) and os.path.exists(os.path.join(base_dir, "test.parquet")):
            print("train.parquet and test.parquet already exist. Skipping generation.")
            return
        else:
            print(f"Error: Could not find original parequet file in {base_dir}")
            return

    # Assuming there's one main file or we concat them
    print(f"Loading data from: {original_files}")
    dfs = [pd.read_parquet(f) for f in original_files]
    df = pd.concat(dfs, ignore_index=True)

    print(f"Total examples: {len(df)}")

    # Check if 'problem' and 'type'/'category' exist
    # (Based on previous conversation, keys might vary, but unsloth loader used 'problem', 'type')

    # Simple random split (10% test)
    # We use a fixed seed for reproducibility
    validation_split = 0.1

    # Shuffle indices
    np.random.seed(42)
    indices = np.random.permutation(len(df))
    split_point = int(len(df) * (1 - validation_split))

    train_indices = indices[:split_point]
    test_indices = indices[split_point:]

    train_df = df.iloc[train_indices]
    test_df = df.iloc[test_indices]

    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    # Save
    train_path = os.path.join(base_dir, "train.parquet")
    test_path = os.path.join(base_dir, "test.parquet")

    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    print(f"Saved to {train_path} and {test_path}")

if __name__ == "__main__":
    create_splits()
