try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None

from .config import load_config, get_config
from .data_loader import get_dataset

import torch
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import numpy as np
import time
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm
import argparse
import yaml
import gc # Added for memory cleanup

MODEL_PATH_DEFAULT = "fine_tunings/llama3:8b" # Or "outputs/checkpoint-60" if testing midway

def evaluate_single_model(model_path, dataset_override=None):
    print(f"\nProcessing Model: {model_path}")

    try:
        conf = load_config(model_path=model_path)
        # Allow overriding dataset even for fine-tuned models if needed
        if dataset_override:
            conf.dataset_path = dataset_override
    except (FileNotFoundError, OSError):
        print(f"Warning: No training_config.json found at {model_path}. Assuming Base Model evaluation or incomplete run.")
        # Only proceed if it looks like a model
        if not dataset_override:
            # Default fallback if not provided
            dataset_override = "qwedsacf/competition_math"
            print(f"Using default dataset: {dataset_override}")

        conf = get_config(
            model_name=model_path,
            dataset_path=dataset_override,
            run_name="base_model_eval",
            num_epochs=0
        )

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length = conf.max_seq_len,
            load_in_4bit = conf.load_in_4bit,
            dtype = None,
        )
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return

    FastLanguageModel.for_inference(model)

    _, _, test_dataset = get_dataset(config= conf)

    y_true, y_pred, tokens_per_sec = [], [], []

    pbar = tqdm(test_dataset, desc=f"Evaluating {os.path.basename(model_path)}", unit="sample")
    for x in pbar:
        prompt, label = x["prompt"], x["type"]

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,  # Class labels are short (1-3 tokens)
                do_sample=False,    # Greedy decoding is faster
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        end = time.time()

        # Robust decoding: decode ONLY the new tokens
        new_tokens = outputs[0, inputs.input_ids.shape[1]:]
        raw_answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Clean up: Take only the first line/segment (model might hallucinate next input)
        generated_answer = raw_answer.split('\n')[0].strip()
        # Also handle commonly hallucinated headers if they appear on the same line
        for stop_word in ["Input:", "Instruction:", "Output:"]:
            if stop_word in generated_answer:
                generated_answer = generated_answer.split(stop_word)[0].strip()

        # Debug first few samples
        if len(y_true) < 5:
            print(f"\n=== Sample {len(y_true)} ===")
            print(f"Label (y_true): '{label}'")
            print(f"Raw answer: '{raw_answer[:100]}...'")
            print(f"Cleaned (y_pred): '{generated_answer}'")
            print(f"Match: {label == generated_answer}")

        y_true.append(label)
        y_pred.append(generated_answer)
        n_new_tokens = len(new_tokens)
        current_tps = n_new_tokens / (end - start)
        tokens_per_sec.append(current_tps)

        # Track generated length
        if 'generated_lengths' not in locals(): generated_lengths = []
        generated_lengths.append(n_new_tokens)

        # Calculate running accuracy
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        running_acc = correct / len(y_true) * 100

        # Update progress bar with useful info
        pbar.set_postfix({
            "acc": f"{running_acc:.1f}%",
            "tok/s": f"{current_tps:.1f}",
            "pred": generated_answer[:15] + "..." if len(generated_answer) > 15 else generated_answer
        })

    # Get valid classes for the report (only the ones in y_true)
    valid_classes = sorted(set(y_true))
    report_dict = classification_report(y_true, y_pred, labels=valid_classes, zero_division=0, output_dict=True)
    print(classification_report(y_true, y_pred, labels=valid_classes, zero_division=0))
    print(f"Average Tokens/Sec: {np.mean(tokens_per_sec):.2f}")
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    _, _, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    avg_tokens_sec = np.mean(tokens_per_sec)

    # 2. Prepare Data - include per-class metrics
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path,
        "model_name": conf.model_name,
        "run_name": conf.run_name,
        "dataset": conf.dataset_path,
        "test_size": len(y_true),
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "tokens_per_sec": avg_tokens_sec,
        "num_epochs": conf.num_epochs,
        "batch_size": conf.batch_size,
        "learning_rate": conf.learning_rate,
        "lora_rank": conf.lora_rank,
        "lora_alpha": conf.lora_alpha,
    }

    # Add per-class metrics
    for class_name in valid_classes:
        if class_name in report_dict:
            results[f"{class_name}_precision"] = report_dict[class_name]["precision"]
            results[f"{class_name}_recall"] = report_dict[class_name]["recall"]
            results[f"{class_name}_f1"] = report_dict[class_name]["f1-score"]
            results[f"{class_name}_support"] = report_dict[class_name]["support"]

    # 3. Save to CSV
    csv_file = "experiment_results/classification/0_supervised_baselines_results/experiments.csv"

    # Check if CSV exists, if not create it with headers
    if not os.path.exists(csv_file):
        os.makedirs(os.path.dirname(csv_file) or '.', exist_ok=True)
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "model_path", "model_name", "run_name", "dataset", "test_size", "accuracy", "precision_weighted", "recall_weighted", "f1_weighted", "tokens_per_sec", "num_epochs", "batch_size", "learning_rate", "lora_rank", "lora_alpha"] + [f"{c}_precision" for c in valid_classes] + [f"{c}_recall" for c in valid_classes] + [f"{c}_f1" for c in valid_classes] + [f"{c}_support" for c in valid_classes])

    df = pd.DataFrame([results])
    df.to_csv(csv_file, mode='a', header=False, index=False)

    print(f"Results saved to {csv_file}")

    # 4. Save to Central YAML (Requested Format)
    yaml_path = "experiment_results/classification/0_supervised_baselines_results/evaluation_results.yaml"

    # Calculate average generated tokens
    avg_completion_tokens = np.mean(generated_lengths) if 'generated_lengths' in locals() and generated_lengths else 0

    # Construct record
    yaml_record = {
        "Model": conf.model_name,
        "Dataset": conf.dataset_path,
        "Finetune": "generative classification finetune",
        "FineTuned_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1_Weighted": float(f1),
        "F1_Macro": float(f1_macro),
        "Avg_Completion_Tokens": float(avg_completion_tokens),
        "Avg_Latency_Seconds": float(1.0 / avg_tokens_sec * avg_completion_tokens) if avg_tokens_sec > 0 else 0.0,
        "Task": "FineTuned classification"
    }

    # Load existing to append
    existing_data = []
    if os.path.exists(yaml_path):
        try:
            with open(yaml_path, "r") as f:
                existing_data = yaml.safe_load(f) or []
                if not isinstance(existing_data, list):
                    existing_data = [existing_data]
        except Exception as e:
            print(f"Warning: Could not read existing YAML: {e}")

    existing_data.append(yaml_record)

    # Ensure dir exists
    os.makedirs(os.path.dirname(yaml_path) if os.path.dirname(yaml_path) else '.', exist_ok=True)

    with open(yaml_path, "w") as f:
        yaml.dump(existing_data, f, sort_keys=False)

    print(f"Results appended to {yaml_path}")

    # 5. Save to Local Model Directory YAML
    if model_path and os.path.isdir(model_path):
        local_yaml_path = os.path.join(model_path, "evaluation_results.yaml")

        local_data = []
        if os.path.exists(local_yaml_path):
            try:
                with open(local_yaml_path, "r") as f:
                    local_data = yaml.safe_load(f) or []
                    if not isinstance(local_data, list):
                        local_data = [local_data]
            except Exception as e:
                print(f"Warning: Could not read local YAML: {e}")

        local_data.append(yaml_record)

        with open(local_yaml_path, "w") as f:
            yaml.dump(local_data, f, sort_keys=False)

        print(f"Results appended to {local_yaml_path}")

    # Cleanup memory
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleared.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=None, help="Path to single model. If None, batches all in fine_tunings/unsloth")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset path (required for base models)")
    args = parser.parse_args()

    if args.model_path:
        evaluate_single_model(args.model_path, args.dataset)
    else:
        # Batch Mode
        base_dir = "experiment_results/classification/0_supervised_baselines_results/unsloth"
        if not os.path.exists(base_dir):
            print(f"Error: Directory {base_dir} does not exist.")
            exit(1)

        print(f"Searching for models in {base_dir}...")

        # Get all subdirectories
        subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

        if not subdirs:
            print("No models found.")
            exit(0)

        for model_path in subdirs:
            evaluate_single_model(model_path, args.dataset)
