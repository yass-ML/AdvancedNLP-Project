from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
import pandas as pd
import json
import argparse
import os

def train_ner_dpo(dataset_path="ner_dpo_pairs.jsonl", output_model="dpo_selector_model_ner", epochs=3, batch_size=16):
    print(f"Loading DPO pairs from {dataset_path}...")
    train_examples = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            query = data['query']
            chosen = data['chosen']
            rejected = data['rejected']

            # For CrossEncoder DPO, we simply treat (query, chosen) as label 1 and (query, rejected) as label 0?
            # Wait, standard CrossEncoder is usually classification/regression.
            # DPO implies preference optimization.
            # Usually we use a specialized loss or just train with BCE (1 for chosen, 0 for rejected).
            # The previous math DPO file seemed to use a standard CrossEncoder.
            # Let's check `src/train_dpo_selector.py` implementation if possible,
            # but I will stick to what works:
            # Train model to output high score for (query, chosen) and low for (query, rejected).
            # We can pass them as separate examples with labels 1.0 and 0.0.

            train_examples.append(InputExample(texts=[query, chosen], label=1.0))
            train_examples.append(InputExample(texts=[query, rejected], label=0.0))

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)

    print("Initializing CrossEncoder...")
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', num_labels=1)

    model.fit(train_dataloader=train_dataloader, epochs=epochs, warmup_steps=100)

    print(f"Saving model to {output_model}...")
    model.save(output_model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="ner_dpo_pairs.jsonl")
    parser.add_argument("--output", default="dpo_selector_model_ner")
    parser.add_argument("--epochs", type=int, default=1) # Low epochs for demo speed
    args = parser.parse_args()

    train_ner_dpo(dataset_path=args.pairs, output_model=args.output, epochs=args.epochs)
