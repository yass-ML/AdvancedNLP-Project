import random
import pandas as pd
import numpy as np
import os
import pickle
from typing import List, Dict, Any

try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer, CrossEncoder, util
    import torch
except ImportError:
    pass

class ShotSelector:
    def __init__(self, dataset: pd.DataFrame, method: str = "random", k: int = 3, device: str = "cpu", dpo_model_path: str = "dpo_selector_model"):
        """
        Initialize the ShotSelector.
        
        Args:
            dataset: Pandas DataFrame containing 'problem' and 'solution' columns.
            method: Strategy to use ('random', 'semantic', 'lexical', 'cross_encoder', 'dpo').
            k: Number of examples to retrieve.
            device: Device to run models on ('cpu' or 'cuda').
            dpo_model_path: Path to the locally trained DPO selector model.
        """
        self.dataset = dataset.reset_index(drop=True)
        self.method = method
        self.k = k
        self.device = device
        self.dpo_model_path = dpo_model_path
        
        if 'problem' not in self.dataset.columns:
            raise ValueError("Dataset must contain 'problem' column")
        
        self.problems = self.dataset['problem'].tolist()
        self._initialize_models()

    def _initialize_models(self):
        # --- Shared Embedding Logic for Semantic/Cross-Encoder/DPO ---
        if self.method in ["semantic", "cross_encoder", "dpo"]:
            print("Loading Semantic Model (all-MiniLM-L6-v2) for retrieval...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            
            embeddings_path = "embeddings/all-MiniLM-L6-v2-emb.pkl"
            embeddings_path = "embeddings/all-MiniLM-L6-v2-emb.pkl"
            if os.path.exists(embeddings_path):
                print(f"Loading cached embeddings from {embeddings_path}...")
                self.corpus_embeddings = self.load_embeddings(embeddings_path)
                
                # Validation
                if len(self.corpus_embeddings) != len(self.problems):
                    print(f"Warning: Cached embeddings size ({len(self.corpus_embeddings)}) does not match dataset size ({len(self.problems)}). Re-encoding...")
                    self.corpus_embeddings = self.embedder.encode(
                        self.problems, 
                        convert_to_tensor=True, 
                        show_progress_bar=True,
                        device=self.device
                    )
                    self.save_embeddings(self.corpus_embeddings, embeddings_path)
            else:
                print("Encoding dataset...")
                self.corpus_embeddings = self.embedder.encode(
                    self.problems, 
                    convert_to_tensor=True, 
                    show_progress_bar=True,
                    device=self.device
                )
                self.save_embeddings(self.corpus_embeddings, embeddings_path)

        # --- Strategy Specific Initializations ---
        if self.method == "random":
            pass
            
        elif self.method == "lexical":
            print("Tokenizing corpus for BM25...")
            tokenized_corpus = [doc.split(" ") for doc in self.problems]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
        elif self.method == "cross_encoder":
            print("Loading Standard Cross-Encoder (ms-marco-MiniLM-L-6-v2)...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)

        elif self.method == "dpo":
            print(f"Loading DPO-Trained Selector from: {self.dpo_model_path}...")
            # The CrossEncoder wrapper can load local HF models if they have config.json
            if os.path.exists(self.dpo_model_path):
                self.cross_encoder = CrossEncoder(self.dpo_model_path, device=self.device, num_labels=1)
            else:
                raise FileNotFoundError(f"DPO Model not found at {self.dpo_model_path}. Did you run the training script?")
    
    def save_embeddings(self, embeddings, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if torch.is_tensor(embeddings):
            embeddings_np = embeddings.cpu().numpy()
        else:
            embeddings_np = embeddings
        with open(path, 'wb') as f:
            pickle.dump(embeddings_np, f)
    
    def load_embeddings(self, path: str):
        with open(path, 'rb') as f:
            embeddings_np = pickle.load(f)
        return torch.tensor(embeddings_np, device=self.device)

    def select(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        if k is None: k = self.k

        if self.method == "dpo":
            fetch_k = 50
            query_embedding = self.embedder.encode(query, convert_to_tensor=True, device=self.device)
            
            hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=fetch_k)[0]
            
            candidate_indices = []
            semantic_scores = []
            
            for hit in hits:
                idx = hit['corpus_id']
                if self.problems[idx].strip() != query.strip():
                    candidate_indices.append(idx)
                    semantic_scores.append(hit['score'])
            
            if not candidate_indices: return []

            # 2. Calculate DPO Probabilities
            candidate_problems = [self.problems[idx] for idx in candidate_indices]
            model_inputs = [[query, prob] for prob in candidate_problems]
            
            dpo_logits = self.cross_encoder.predict(model_inputs)
            dpo_probs = 1 / (1 + np.exp(-dpo_logits)) 
            
            # 3. HYBRID SCORING (The Fix)
            # Alpha determines the balance.
            # 0.4 means "40% Semantic Similarity + 60% Label Correctness"
            alpha = 0.6

            final_scores = []
            for sem_score, dpo_prob in zip(semantic_scores, dpo_probs):
                hybrid_score = (alpha * sem_score) + ((1 - alpha) * dpo_prob)
                final_scores.append(hybrid_score)
            
            scored_candidates = list(zip(candidate_indices, final_scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            top_indices = [x[0] for x in scored_candidates[:k]]
            return self.dataset.iloc[top_indices].to_dict('records')
            
        if self.method == "random":
            indices = random.sample(range(len(self.dataset)), k)
            return self.dataset.iloc[indices].to_dict('records')
            
        elif self.method == "semantic":
            query_embedding = self.embedder.encode(query, convert_to_tensor=True, device=self.device)
            hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=k)[0]
            indices = [hit['corpus_id'] for hit in hits]
            return self.dataset.iloc[indices].to_dict('records')
            
        elif self.method == "lexical":
            tokenized_query = query.split(" ")
            scores = self.bm25.get_scores(tokenized_query)
            top_n_indices = np.argsort(scores)[::-1][:k]
            return self.dataset.iloc[top_n_indices].to_dict('records')
            
        elif self.method in ["cross_encoder", "dpo"]:
            # 1. Retrieve Candidates (High Recall: fetch 50 candidates)
            fetch_k = min(50, len(self.dataset))
            query_embedding = self.embedder.encode(query, convert_to_tensor=True, device=self.device)
            hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=fetch_k)[0]
            
            # 2. Re-rank with CrossEncoder (DPO or Standard)
            candidate_indices = [hit['corpus_id'] for hit in hits]
            
            # Filter out the query itself if it exists in the dataset (Data Leakage Protection)
            # We check text similarity string-wise to be safe
            filtered_indices = []
            for idx in candidate_indices:
                if self.problems[idx].strip() != query.strip():
                    filtered_indices.append(idx)
            
            if not filtered_indices: return [] # Handle edge case
            
            candidate_problems = [self.problems[idx] for idx in filtered_indices]
            model_inputs = [[query, prob] for prob in candidate_problems]
            
            # Predict scores
            cross_scores = self.cross_encoder.predict(model_inputs)
            
            # Sort
            scored_candidates = list(zip(filtered_indices, cross_scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Select top k
            top_indices = [x[0] for x in scored_candidates[:k]]
            return self.dataset.iloc[top_indices].to_dict('records')
        
        else:
            raise ValueError(f"Unknown method: {self.method}")