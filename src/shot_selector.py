import random
import pandas as pd
import numpy as np
from typing import List, Dict, Any

# Conditional imports to avoid errors if dependencies aren't ready yet (though they should be)
try:
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer, CrossEncoder, util
    import torch
except ImportError:
    pass

class ShotSelector:
    def __init__(self, dataset: pd.DataFrame, method: str = "random", k: int = 3, device: str = "cpu"):
        """
        Initialize the ShotSelector.
        
        Args:
            dataset: Pandas DataFrame containing 'problem' and 'solution' columns.
            method: Strategy to use ('random', 'semantic', 'lexical', 'cross_encoder').
            k: Number of examples to retrieve.
            device: Device to run models on ('cpu' or 'cuda').
        """
        self.dataset = dataset.reset_index(drop=True)
        self.method = method
        self.k = k
        self.device = device
        
        # Ensure required columns exist
        if 'problem' not in self.dataset.columns:
            raise ValueError("Dataset must contain 'problem' column")
        # Assume 'solution' exists, if not finding it might require inspection, 
        # but for now we proceed. If it's missing, retrieval works but returning solution fails.
        
        self.problems = self.dataset['problem'].tolist()
        
        self._initialize_models()

    def _initialize_models(self):
        if self.method == "random":
            pass
            
        elif self.method == "semantic":
            print("Loading Semantic Model (all-MiniLM-L6-v2)...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            print("Encoding dataset...")
            # Pre-compute embeddings for the whole dataset
            self.corpus_embeddings = self.embedder.encode(
                self.problems, 
                convert_to_tensor=True, 
                show_progress_bar=True,
                device=self.device
            )
            
        elif self.method == "lexical":
            print("Tokenizing corpus for BM25...")
            tokenized_corpus = [doc.split(" ") for doc in self.problems]
            self.bm25 = BM25Okapi(tokenized_corpus)
            
        elif self.method == "cross_encoder":
            # Hybrid approach: Semantic retrieval -> Cross-Encoder Re-ranking
            print("Loading Semantic Model for candidates...")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            print("Encoding dataset...")
            self.corpus_embeddings = self.embedder.encode(
                self.problems, 
                convert_to_tensor=True, 
                show_progress_bar=True,
                device=self.device
            )
            
            print("Loading Cross-Encoder (ms-marco-MiniLM-L-6-v2)...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)

    def select(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Select k examples for the given query.
        """
        if k is None:
            k = self.k
            
        if self.method == "random":
            indices = random.sample(range(len(self.dataset)), k)
            return self.dataset.iloc[indices].to_dict('records')
            
        elif self.method == "semantic":
            query_embedding = self.embedder.encode(query, convert_to_tensor=True, device=self.device)
            # Find top k similar
            hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=k)[0]
            indices = [hit['corpus_id'] for hit in hits]
            return self.dataset.iloc[indices].to_dict('records')
            
        elif self.method == "lexical":
            tokenized_query = query.split(" ")
            # get_top_n returns the documents, not indices. 
            # We need indices to get full records efficiently or we use get_scores.
            scores = self.bm25.get_scores(tokenized_query)
            top_n_indices = np.argsort(scores)[::-1][:k]
            return self.dataset.iloc[top_n_indices].to_dict('records')
            
        elif self.method == "cross_encoder":
            # 1. Retrieve candidates (e.g., 10 * k) using semantic search
            num_candidates = min(k * 10, len(self.dataset))
            query_embedding = self.embedder.encode(query, convert_to_tensor=True, device=self.device)
            hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=num_candidates)[0]
            
            # 2. Re-rank with CrossEncoder
            candidate_indices = [hit['corpus_id'] for hit in hits]
            candidate_problems = [self.problems[idx] for idx in candidate_indices]
            model_inputs = [[query, prob] for prob in candidate_problems]
            
            cross_scores = self.cross_encoder.predict(model_inputs)
            
            # Sort by cross_scores
            # Combine indices and scores
            scored_candidates = list(zip(candidate_indices, cross_scores))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Select top k
            top_indices = [x[0] for x in scored_candidates[:k]]
            return self.dataset.iloc[top_indices].to_dict('records')
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
