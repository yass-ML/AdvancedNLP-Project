try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None

from .config import load_config
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

MODEL_PATH = "fine_tunings/llama3:8b" # Or "outputs/checkpoint-60" if testing midway

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    args = parser.parse_args()
    conf = load_config(model_path=args.model_path)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length = conf.max_seq_len,
        load_in_4bit = conf.load_in_4bit,
        dtype = None,
    )

    FastLanguageModel.for_inference(model)

    _, _, test_dataset = get_dataset(config= conf)
    test_dataset = test_dataset.select(range(min(100, len(test_dataset))))  # Limit to 200 samples
    y_true, y_pred, tokens_per_sec = [], [], []

    pbar = tqdm(test_dataset, desc="Evaluating", unit="sample")
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


        # outputs_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # prediction_text = outputs_decoded[0]
        # generated_answer = prediction_text[len(prompt):].strip()

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
    avg_tokens_sec = np.mean(tokens_per_sec)

    # 2. Prepare Data - include per-class metrics
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": args.model_path,
        "model_name": conf.model_name,
        "run_name": conf.run_name,
        "dataset": conf.dataset_path,
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
    csv_file = "fine_tunings/experiments.csv"
    df = pd.DataFrame([results])

    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)

    print(f"Results saved to {csv_file}")
