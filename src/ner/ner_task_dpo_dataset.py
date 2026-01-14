import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm
import random
import json
import os

class NERDPODatasetPipeline:
    def __init__(self, dataset_path="datasets/few_nerd/train.parquet", embedding_model="all-MiniLM-L6-v2"):
        self.dataset_path = dataset_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(embedding_model, device=self.device)
        self.corpus_embeddings = None
        self.train_data = None

    def load_data(self):
        print(f"Loading NER data from {self.dataset_path}...")
        df = pd.read_parquet(self.dataset_path)
        # Store as list of dicts
        self.train_data = df.to_dict(orient='records')
        print(f"Loaded {len(self.train_data)} examples.")

        # Precompute entity sets for faster comparison
        # entity set = set of unique tags present in the sentence (excluding O)
        for item in self.train_data:
            item['entity_set'] = set([t for t in item['ner_tags'] if t != 'O'])

        print("Encoding corpus...")
        texts = [item['text'] for item in self.train_data]
        self.corpus_embeddings = self.embedder.encode(texts, convert_to_tensor=True, show_progress_bar=True)

    def generate_dpo_pairs(self, output_file, num_samples=1000):
        dpo_pairs = []

        # Sample queries
        indices = list(range(len(self.train_data)))
        random.shuffle(indices)
        selected_indices = indices[:num_samples]

        print("Generating pairs...")
        for idx in tqdm(selected_indices):
            query_item = self.train_data[idx]
            query_text = query_item['text']
            query_entities = query_item['entity_set']

            if not query_entities: # Skip sentences with no entities? Or keep them?
                # If no entities, it's hard to define semantic correctness based on entities.
                # Let's skip empty ones for this experiment logic.
                continue

            query_emb = self.corpus_embeddings[idx]

            # Retrieve neighbors
            hits = util.semantic_search(query_emb, self.corpus_embeddings, top_k=50)[0]

            potential_winners = []
            hard_negatives = []

            for hit in hits:
                cand_idx = hit['corpus_id']
                if cand_idx == idx: continue

                candidate = self.train_data[cand_idx]
                cand_text = candidate['text']
                cand_entities = candidate['entity_set']

                if cand_text == query_text: continue

                # Logic:
                # Winner = Shares at least one entity type (or high Jaccard index of types)
                # Hard Negative = High semantic similarity BUT shares NO entity types (or low overlap)

                intersection = query_entities.intersection(cand_entities)

                if len(intersection) > 0:
                    # It's a potential winner (shares types)
                    # We could rank by how MANY types they share.
                    potential_winners.append(cand_text)
                else:
                    # It's a hard negative (semantically close but different types)
                    hard_negatives.append(cand_text)

            if potential_winners and hard_negatives:
                best_winner = potential_winners[0] # Top semantic match that is relevant

                # Pick top negatives
                for loser in hard_negatives[:3]:
                    dpo_pairs.append({
                        "query": query_text,
                        "chosen": best_winner,
                        "rejected": loser,
                        "entities": list(query_entities)
                    })

        with open(output_file, 'w') as f:
            for entry in dpo_pairs:
                f.write(json.dumps(entry) + '\n')

        print(f"Saved {len(dpo_pairs)} pairs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="ner_dpo_pairs.jsonl")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    pipeline = NERDPODatasetPipeline()
    pipeline.load_data()
    pipeline.generate_dpo_pairs(args.output, args.samples)
