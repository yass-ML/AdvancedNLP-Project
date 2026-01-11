from .config import get_config
from .trainer import train

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model on a dataset")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run (for saving checkpoints)")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., llama3:8b) or path to checkpoint")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--dataset", type=str, default="qwedsacf/competition_math", help="HuggingFace dataset path")

    args = parser.parse_args()

    conf = get_config(
        run_name=args.run_name,
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        dataset_path=args.dataset
    )

    train(conf)
