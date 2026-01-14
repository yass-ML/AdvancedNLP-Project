import os
import pandas as pd
from datasets import load_dataset
import argparse

def load_and_process_few_nerd(output_dir="datasets/few_nerd"):
    """
    Downloads the Few-NERD dataset (supervised split) and saves it as parquet files.
    """
    print("Downloading DFKI-SLT/few-nerd (supervised)...")
    try:
        # Load the supervised split
        dataset = load_dataset("DFKI-SLT/few-nerd", "supervised")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)

    splits = ["train", "validation", "test"] # few-nerd has train, val, test, but check actual keys

    # Check actual keys
    print(f"Available splits: {dataset.keys()}")

    for split in dataset.keys():
        print(f"Processing {split} split...")
        data = []
        for item in dataset[split]:
            # item has 'tokens' (list of str) and 'ner_tags' (list of int/str)
            # The original dataset might have ner_tags as IDs. We need to map them if they are IDs.
            # But usually HuggingFace datasets handle this. Let's check features.

            tokens = item['tokens']
            tags = item['ner_tags']

            # Convert tags to string labels if they are integers
            if hasattr(dataset[split].features['ner_tags'], 'feature'):
                int2str = dataset[split].features['ner_tags'].feature.int2str
                tags = [int2str(t) for t in tags]

            data.append({
                "tokens": tokens,
                "ner_tags": tags,
                "text": " ".join(tokens) # For simple retrieval/viewing
            })

        df = pd.DataFrame(data)
        output_path = os.path.join(output_dir, f"{split}.parquet")
        df.to_parquet(output_path)
        print(f"Saved {len(df)} examples to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="datasets/few_nerd")
    args = parser.parse_args()

    load_and_process_few_nerd(args.output_dir)
