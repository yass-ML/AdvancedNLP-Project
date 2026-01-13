"""
Batch fine-tuning script for running classification training on multiple models.

Usage:
    python -m src.baselines.supervised.classification_head.fine_tune \
        --num_epochs 3 \
        --dataset qwedsacf/competition_math
"""

import argparse
import subprocess
import sys

from .model import _MODEL_MAP


# Base models for classification (non-instruct versions preferred)
CLASSIFICATION_MODELS = {
    # Llama 3
    "llama3:8b": "meta-llama/Meta-Llama-3-8B",
    # "llama3:8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",

    # Mistral
    "mistral:7b": "mistralai/Mistral-7B-v0.3",
    # "mistral:7b-instruct": "mistralai/Mistral-7B-Instruct-v0.3",

    # Gemma
    "gemma:7b": "google/gemma-7b",
    # "gemma:7b-instruct": "google/gemma-7b-it",

    # Phi-3
    "phi3:mini": "microsoft/Phi-3-mini-4k-instruct",

    # Qwen 2 & 2.5
    "qwen2:7b": "Qwen/Qwen2-7B",
    "qwen2:7b-instruct": "Qwen/Qwen2-7B-Instruct",
}


def main():
    parser = argparse.ArgumentParser(
        description="Run classification fine-tuning on multiple models."
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs per model.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="qwedsacf/competition_math",
        help="Dataset to use for training.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to train (default: all in registry)",
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Use LoRA for fine-tuning",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Use 4-bit quantization (QLoRA)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size per device",
    )
    parser.add_argument(
        "--subset_limit",
        type=float,
        default=None,
        help="Limit dataset size for quick testing",
    )
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        default=True,
        help="Use focal loss for class imbalance (default: True)",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter",
    )

    args = parser.parse_args()

    if args.models:
        models = args.models
    else:
        models = list(CLASSIFICATION_MODELS.keys())

    print(f"Will train {len(models)} models: {models}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.num_epochs}")
    print("-" * 50)

    results = {}

    for model in models:
        run_name = f"classify_{model.replace(':', '_')}_{args.dataset.replace('/', '_')}"
        print(f"\n{'='*60}")
        print(f"Starting training for {model}...")
        print(f"Run name: {run_name}")
        print(f"{'='*60}\n")

        cmd = [
            sys.executable, "-m", "src.baselines.supervised.classification_head.train",
            "--run_name", run_name,
            "--model_name", model,
            "--num_epochs", str(args.num_epochs),
            "--dataset", args.dataset,
            "--batch_size", str(args.batch_size),
        ]

        if args.use_lora:
            cmd.append("--use_lora")

        if args.load_in_4bit:
            cmd.append("--load_in_4bit")

        if args.subset_limit:
            cmd.extend(["--subset_limit", str(args.subset_limit)])

        if args.use_focal_loss:
            cmd.append("--use_focal_loss")
            cmd.extend(["--focal_gamma", str(args.focal_gamma)])

        try:
            subprocess.run(cmd, check=True)
            print(f"✓ Successfully trained {model}")
            results[model] = "success"
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to train {model}. Error: {e}")
            results[model] = f"failed: {e}"

    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    for model, status in results.items():
        symbol = "✓" if status == "success" else "✗"
        print(f"  {symbol} {model}: {status}")


if __name__ == "__main__":
    main()
