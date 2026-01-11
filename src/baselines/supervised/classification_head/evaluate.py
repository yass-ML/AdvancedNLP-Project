"""
Evaluation script for Classification Head models.

Loads a fine-tuned classification model and evaluates on test set.

Usage:
    python -m src.baselines.supervised.classification_head.evaluate \
        --model_path fine_tunings/classification_head/gemma_2b_math
"""

import argparse
import json
import time
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from .config import load_config, get_config
from .model import load_model_for_inference, get_model_id
from .data_loader import get_raw_dataset


def evaluate_model(
    model,
    tokenizer,
    test_dataset,
    config,
    batch_size: int = 16,
):
    """
    Evaluate a classification model on the test set.

    Args:
        model: The classification model
        tokenizer: Tokenizer
        test_dataset: Raw test dataset with text and labels
        config: Classification config with label mappings
        batch_size: Batch size for inference

    Returns:
        dict with evaluation results
    """
    model.eval()
    device = next(model.parameters()).device

    all_predictions = []
    all_labels = []
    all_probs = []
    inference_times = []

    # Process in batches
    num_samples = len(test_dataset)

    pbar = tqdm(range(0, num_samples, batch_size), desc="Evaluating")
    for start_idx in pbar:
        end_idx = min(start_idx + batch_size, num_samples)
        batch = test_dataset.select(range(start_idx, end_idx))

        # Get texts and labels - convert to list explicitly
        texts = list(batch[config.text_column])
        labels = [config.label2id[l] for l in batch[config.label_column]]

        # Tokenize
        inputs = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_len,
            padding=True,
            return_tensors="pt",
        ).to(device)

        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()

        # Get predictions (convert to float32 for numpy compatibility)
        logits = outputs.logits.float()
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(logits, dim=-1)

        all_predictions.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels)
        all_probs.extend(probs.cpu().numpy().tolist())
        inference_times.append(end_time - start_time)

    # Convert to numpy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average="weighted", zero_division=0
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )

    # Timing stats
    total_time = sum(inference_times)
    avg_time_per_sample = total_time / num_samples
    samples_per_second = num_samples / total_time

    # Ensure id2label uses integer keys (JSON loads them as strings)
    id2label = {int(k): v for k, v in config.id2label.items()}

    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "num_samples": num_samples,
        "total_inference_time_seconds": float(total_time),
        "avg_time_per_sample_ms": float(avg_time_per_sample * 1000),
        "samples_per_second": float(samples_per_second),
        "per_class_metrics": {
            id2label[i]: {
                "precision": float(precision_per_class[i]),
                "recall": float(recall_per_class[i]),
                "f1": float(f1_per_class[i]),
            }
            for i in range(len(id2label))
        },
    }

    return results, all_predictions, all_labels, id2label


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned classification model"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset path (uses training config if not provided)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save evaluation results JSON",
    )

    args = parser.parse_args()

    # Load config
    try:
        config = load_config(args.model_path)
        print(f"Loaded config from {args.model_path}")

        if args.dataset:
            config.dataset_path = args.dataset

    except FileNotFoundError:
        print(f"No config found at {args.model_path}. Creating default config.")
        if args.dataset is None:
            args.dataset = "qwedsacf/competition_math"
        config = get_config(
            model_name=args.model_path,
            dataset_path=args.dataset,
        )

    # Load model
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_for_inference(args.model_path, config)

    # Load test dataset
    print(f"Loading dataset: {config.dataset_path}")
    _, _, test_dataset, label2id, id2label = get_raw_dataset(config)

    # Update config with label mappings if not present
    if not config.label2id:
        config.label2id = label2id
        config.id2label = id2label
        config.num_labels = len(label2id)

    print(f"Test set size: {len(test_dataset)}")
    print(f"Number of classes: {len(config.label2id)}")

    # Evaluate
    print("\nRunning evaluation...")
    results, predictions, labels, id2label = evaluate_model(
        model, tokenizer, test_dataset, config, args.batch_size
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"\nInference Speed: {results['samples_per_second']:.2f} samples/sec")
    print(f"Avg time per sample: {results['avg_time_per_sample_ms']:.2f} ms")

    # Detailed classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        labels, predictions,
        target_names=[id2label[i] for i in sorted(id2label.keys())],
        digits=4,
    )
    print(report)

    # Confusion matrix
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(labels, predictions)
    class_names = [id2label[i] for i in sorted(id2label.keys())]

    # Print header
    header = "True\\Pred".ljust(20) + " ".join([name[:8].ljust(10) for name in class_names])
    print(header)
    print("-" * len(header))

    for i, row in enumerate(cm):
        row_str = class_names[i][:18].ljust(20) + " ".join([str(val).ljust(10) for val in row])
        print(row_str)

    # Save results
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = Path(args.model_path) / "evaluation_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Also save the classification report
    report_path = output_path.parent / "evaluation_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")


if __name__ == "__main__":
    main()
