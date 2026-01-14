"""
Pipeline to generate a relevance-optimized dataset for DPO training using semantic search and sentence embeddings.
"""

import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import random
import json
import os

class RelevanceDPOPipeline:
    def __init__(self, model_name="phi3:mini", embedding_model="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading Embedding Model: {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        self.corpus_embeddings = None
        self.train_data = None

    def load_local_parquet(self, file_path):
        print(f"Loading local Parquet file: {file_path}...")
        try:
            df = pd.read_parquet(file_path)
            # Reset index to ensure it matches the list indices later
            df.reset_index(drop=True, inplace=True)
        except Exception as e:
            raise FileNotFoundError(f"Could not read file. Error: {e}")

        if 'type' in df.columns:
            df = df.rename(columns={'problem': 'text', 'type': 'label'})
        elif 'category' in df.columns:
            df = df.rename(columns={'problem': 'text', 'category': 'label'})

        self.df_raw = df
        self.train_data = df.to_dict(orient='records')
        print(f"Loaded {len(self.train_data)} examples.")

        print("Encoding corpus for retrieval...")
        texts = [item['text'] for item in self.train_data]
        self.corpus_embeddings = self.embedder.encode(texts, convert_to_tensor=True, show_progress_bar=True)

    def generate_relevance_dataset(self, output_file, num_samples=1000, batch_size=32):
        dpo_pairs = []

        print(f"Sampling {num_samples} queries...")
        grouped = self.df_raw.groupby('label')
        samples_per_cat = max(1, num_samples // len(grouped))

        selected_indices = []
        for label, group in grouped:
            n = min(len(group), samples_per_cat)
            selected_indices.extend(group.sample(n, random_state=42).index.tolist())
        random.shuffle(selected_indices)

        print(f"Generating Relevance-Optimized Pairs...")

        for i in tqdm(range(0, len(selected_indices), batch_size)):
            batch_indices = selected_indices[i : i + batch_size]
            batch_queries = [self.train_data[idx]['text'] for idx in batch_indices]

            query_embs = self.embedder.encode(batch_queries, convert_to_tensor=True)
            batch_hits = util.semantic_search(query_embs, self.corpus_embeddings, top_k=50)

            for j, hits in enumerate(batch_hits):
                query_idx = batch_indices[j]
                query_text = batch_queries[j]
                true_label = self.train_data[query_idx]['label']

                potential_winners = []
                hard_negatives = []

                for hit in hits:
                    cand_idx = hit['corpus_id']
                    if cand_idx == query_idx:
                        continue

                    candidate = self.train_data[cand_idx]
                    cand_text = candidate['text']
                    cand_label = candidate.get('label', 'Unknown')

                    if cand_text.strip() == query_text.strip():
                        continue

                    if cand_label == true_label:
                        potential_winners.append(cand_text)
                    else:
                        hard_negatives.append(cand_text)

                # 3. Form Pairs: Best Winner vs. Hardest Losers
                # We want the "Winner" to be the Most Similar example that is correct.
                # We want the "Loser" to be a High Similarity example that is wrong.

                if not potential_winners or not hard_negatives:
                    continue

                # Strategy: Pick the #1 Best Winner (Most similar correct example)
                best_winner = potential_winners[0]

                # Strategy: Pair it against the top 3 Hard Negatives
                # This teaches: "Even though this Negative looks similar, the Winner is BETTER."
                for loser in hard_negatives[:3]:
                    dpo_pairs.append({
                        "query": query_text,
                        "chosen": best_winner, # High Sim + Correct Label
                        "rejected": loser,     # High Sim + Wrong Label
                        "label_type": true_label
                    })

        with open(output_file, 'w') as f:
            for entry in dpo_pairs:
                f.write(json.dumps(entry) + '\n')

        print(f"Saved {len(dpo_pairs)} relevance-optimized pairs to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="math_dpo_relevance.jsonl")
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--model", type=str, default="phi3:mini")

    args = parser.parse_args()

    pipeline = RelevanceDPOPipeline(model_name=args.model)
    if os.path.exists(args.dataset_path):
        pipeline.load_local_parquet(args.dataset_path)
        pipeline.generate_relevance_dataset(args.output, num_samples=args.samples)
    else:
        print(f"Error: File not found at {args.dataset_path}")
