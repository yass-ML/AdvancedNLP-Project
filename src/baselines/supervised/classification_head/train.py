"""
CLI entry point for training classification models.

Usage:
    python -m src.baselines.supervised.classification_head.train \
        --run_name gemma_2b_math_classification \
        --model_name gemma:2b \
        --dataset qwedsacf/competition_math \
        --num_epochs 3 \
        --use_lora
"""

import argparse

from .config import get_config
from .trainer import train


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a model with classification head"
    )

    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Name of the run (for saving checkpoints)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (e.g., gemma:2b, llama3:8b) or HuggingFace model ID",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="qwedsacf/competition_math",
        help="HuggingFace dataset path",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="problem",
        help="Column name containing the input text",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="type",
        help="Column name containing the labels",
    )
    parser.add_argument(
        "--subset_limit",
        type=float,
        default=None,
        help="Limit dataset size (float for fraction, int for count)",
    )

    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for parameter-efficient fine-tuning",
    )
    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )

    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Use 4-bit quantization (QLoRA)",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Use 8-bit quantization",
    )

    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        help="Use focal loss to handle class imbalance",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter (higher = more focus on hard examples)",
    )
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable class weights in focal loss",
    )
    parser.add_argument(
        "--class_weight_method",
        type=str,
        default="effective",
        choices=["inverse", "inverse_sqrt", "effective"],
        help="Method for computing class weights",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiment_results/classification/0_supervised_baselines_results/classification_head",
        help="Base output directory",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    use_lora = True
    if args.no_lora:
        use_lora = False
    elif args.use_lora:
        use_lora = True

    use_class_weights = not args.no_class_weights

    config = get_config(
        run_name=args.run_name,
        model_name=args.model_name,
        dataset_path=args.dataset,
        text_column=args.text_column,
        label_column=args.label_column,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len,
        use_lora=use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        output_dir=args.output_dir,
        subset_limit=args.subset_limit,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        use_class_weights=use_class_weights,
        class_weight_method=args.class_weight_method,
    )

    train(config, resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
