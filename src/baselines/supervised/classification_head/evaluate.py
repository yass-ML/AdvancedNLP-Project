"""
Evaluation script for Classification Head models.

Loads a fine-tuned classification model and evaluates on test set.

Usage:
    python -m src.baselines.supervised.classification_head.evaluate \
        --model_path fine_tunings/classification_head/gemma_2b_math

    # Evaluate ALL models in fine_tunings/classification_head
    python -m src.baselines.supervised.classification_head.evaluate
"""

import argparse
import json
import time
import os
from pathlib import Path
import yaml
import gc  # Added for memory cleanup

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    f1_score,
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

    num_samples = len(test_dataset)

    pbar = tqdm(range(0, num_samples, batch_size), desc="Evaluating")
    for start_idx in pbar:
        end_idx = min(start_idx + batch_size, num_samples)
        batch = test_dataset.select(range(start_idx, end_idx))

        texts = list(batch[config.text_column])
        labels = [config.label2id[l] for l in batch[config.label_column]]

        inputs = tokenizer(
            texts,
            truncation=True,
            max_length=config.max_seq_len,
            padding=True,
            return_tensors="pt",
        ).to(device)

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

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average="weighted", zero_division=0
    )
    f1_macro = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )

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
        "f1_macro": float(f1_macro),
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


def evaluate_single_model(model_path, dataset_override=None, batch_size=16, output_file=None):
    """
    Evaluates a single model at the specified path.
    """
    print(f"\nProcessing Model: {model_path}")

    try:
        config = load_config(model_path)
        print(f"Loaded config from {model_path}")

        if dataset_override:
            config.dataset_path = dataset_override

    except FileNotFoundError:
        print(f"No config found at {model_path}. Creating default config.")
        if dataset_override is None:
            dataset_override = "qwedsacf/competition_math"
        config = get_config(
            model_name=model_path,
            dataset_path=dataset_override,
        )

    print(f"Loading model from {model_path}...")
    try:
        model, tokenizer = load_model_for_inference(model_path, config)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    print(f"Loading dataset: {config.dataset_path}")
    _, _, test_dataset, label2id, id2label = get_raw_dataset(config)

    if not config.label2id:
        config.label2id = label2id
        config.id2label = id2label
        config.num_labels = len(label2id)

    print(f"Test set size: {len(test_dataset)}")
    print(f"Number of classes: {len(config.label2id)}")

    print("\nRunning evaluation...")
    results, predictions, labels, id2label = evaluate_model(
        model, tokenizer, test_dataset, config, batch_size
    )

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS: {model_path}")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 Score:  {results['f1']:.4f}")
    print(f"\nInference Speed: {results['samples_per_second']:.2f} samples/sec")
    print(f"Avg time per sample: {results['avg_time_per_sample_ms']:.2f} ms")

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        labels, predictions,
        target_names=[id2label[i] for i in sorted(id2label.keys())],
        digits=4,
    )
    print(report)

    print("\n" + "=" * 60)
    print("CONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(labels, predictions)
    class_names = [id2label[i] for i in sorted(id2label.keys())]

    header = "True\\Pred".ljust(20) + " ".join([name[:8].ljust(10) for name in class_names])
    print(header)
    print("-" * len(header))

    for i, row in enumerate(cm):
        row_str = class_names[i][:18].ljust(20) + " ".join([str(val).ljust(10) for val in row])
        print(row_str)

    if output_file:
        output_path = Path(output_file)
    else:
        output_path = Path(model_path) / "evaluation_results.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    report_path = output_path.parent / "evaluation_classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

    yaml_path = output_path.parent / "evaluation_results.yaml"

    yaml_record = {
        "Model": config.model_name,
        "Dataset": config.dataset_path,
        "Finetune": "classification head classification finetune",
        "FineTuned_modules": config.target_modules + ["score"] if config.use_lora else "all",
        "Accuracy": float(results["accuracy"]),
        "Precision": float(results["precision"]),
        "Recall": float(results["recall"]),
        "F1_Weighted": float(results["f1"]),
        "F1_Macro": float(results["f1_macro"]),
        "Avg_Completion_Tokens": "N/A", # Not applicable for classification head
        "Avg_Latency_Seconds": float(results["avg_time_per_sample_ms"] / 1000),
        "Task": "FineTuned classification"
    }

    existing_data = []
    if yaml_path.exists():
        try:
            with open(yaml_path, "r") as f:
                existing_data = yaml.safe_load(f) or []
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except Exception as e:
            print(f"Warning: Could not read existing YAML: {e}")

    existing_data.append(yaml_record)

    with open(yaml_path, "w") as f:
        yaml.dump(existing_data, f, sort_keys=False)

    print(f"Results appended to {yaml_path}")

    central_yaml_path = Path("fine_tunings/evaluation_results.yaml")

    central_data = []
    if central_yaml_path.exists():
        try:
            with open(central_yaml_path, "r") as f:
                central_data = yaml.safe_load(f) or []
                if not isinstance(central_data, list):
                    central_data = [central_data]
        except Exception as e:
            print(f"Warning: Could not read central YAML: {e}")

    central_data.append(yaml_record)

    central_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    with open(central_yaml_path, "w") as f:
        yaml.dump(central_data, f, sort_keys=False)

    print(f"Results appended to {central_yaml_path}")

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleared.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned classification model(s)"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the fine-tuned model. If not provided, scans fine_tunings/classification_head",
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
        help="Path to save evaluation results JSON (only specific when single model run)",
    )

    args = parser.parse_args()

    if args.model_path:
        evaluate_single_model(args.model_path, args.dataset, args.batch_size, args.output_file)
    else:
        base_dir = Path("fine_tunings/classification_head")
        if not base_dir.exists():
            print(f"Error: Directory {base_dir} does not exist.")
            return

        print(f"Searching for models in {base_dir}...")

        model_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])

        if not model_dirs:
            print("No models found.")
            return

        for model_path in model_dirs:
            evaluate_single_model(str(model_path), args.dataset, args.batch_size)

if __name__ == "__main__":
    main()
