from IPython.lib.pretty import MAX_SEQ_LENGTH
try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None

from .config import get_config
from .data_loader import get_dataset

import torch
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import numpy as np
import time
import pandas as pd
import os
from datetime import datetime

MODEL_PATH = "lora_model" # Or "outputs/checkpoint-60" if testing midway

if __name__ == "__main__":
    conf = get_config()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length = conf.max_seq_len,
        load_in_4bit = conf.load_in_4bit,
        dtype = None,
    )

    FastLanguageModel.for_inference(model)

    _, test_dataset = get_dataset(config= conf)
    y_true, y_pred, tokens_per_sec = [], [], []
    for x in test_dataset:
        prompt, label = x["prompt"], x["type"]

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, use_cache=True)
        end = time.time()


        outputs_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        prediction_text = outputs_decoded[0]
        generated_answer = prediction_text[len(prompt):].strip()

        y_true.append(label)
        y_pred.append(generated_answer)
        n_new_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        tokens_per_sec.append(n_new_tokens / (end - start))

    print(classification_report(y_true, y_pred))
    print(f"Average Tokens/Sec: {np.mean(tokens_per_sec):.2f}")
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    avg_tokens_sec = np.mean(tokens_per_sec)

    # 2. Prepare Data
    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": conf.model_name,
        "dataset": conf.dataset_path,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tokens_per_sec": avg_tokens_sec,
        "batch_size": conf.batch_size,
        "lora_rank": conf.lora_rank
    }

    # 3. Save to CSV
    csv_file = "experiments.csv"
    df = pd.DataFrame([results])

    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)

    print(f"Results saved to {csv_file}")
