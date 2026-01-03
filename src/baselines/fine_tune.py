import argparse
import subprocess
import sys
from .supervised.model_registry import _MODEL_MAP

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fine-tuning on all models in the registry.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs per model.")
    parser.add_argument("--dataset", type=str, default="qwedsacf/competition_math", help="Dataset to use for training.")
    parser.add_argument("--include_instruct", action="store_true", help="If set, include instruct models in fine-tuning.")

    args = parser.parse_args()

    # Example: looping through your models
    models = _MODEL_MAP.keys()

    for model in models:
        if "instruct" in model and not args.include_instruct:
            continue

        print(f"Starting training for {model}...")

        cmd = [
            sys.executable, "-m", "src.baselines.supervised.train",
            "--run_name", f"finetune_{model.replace(':', '_')}",
            "--model_name", model,
            "--num_epochs", str(args.num_epochs),
            "--dataset", args.dataset
        ]

        # Execute the command
        try:
            subprocess.run(cmd, check=True)
            print(f"Successfully trained {model}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to train {model}. Error: {e}")
