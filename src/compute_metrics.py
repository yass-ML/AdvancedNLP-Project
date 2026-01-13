import pandas as pd
from sklearn.metrics import f1_score
import requests
import argparse
import os
import glob
from shot_selector import ShotSelector

class MetricsPipeline:
    def __init__(self, model_name="llama3:8b", dataset_path="datasets/competition_math", 
                 selector_strategy="random", k_shots=0, dpo_model_path="dpo_selector_model"):
        """
        Initializes the pipeline.
        
        Args:
            model_name: The Ollama model tag.
            dataset_path: Path to the dataset folder.
            selector_strategy: 'random', 'semantic', 'lexical', 'cross_encoder', or 'dpo'.
            k_shots: Number of examples to include in the prompt.
            dpo_model_path: Path to the trained DPO Cross-Encoder (only used if strategy='dpo').
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.selector_strategy = selector_strategy
        self.k_shots = k_shots
        self.dpo_model_path = dpo_model_path  # Store the path
        self.selector = None
        self.base_url = "http://localhost:11434/api/generate"
        self.categories = ["Algebra", "Counting & Probability", "Geometry", "Intermediate Algebra", "Number Theory", "Prealgebra", "Precalculus"]

    def load_data(self):
        """Loads the competition_math dataset from parquet files."""
        search_path = os.path.join(self.dataset_path, "**", "*.parquet")
        files = glob.glob(search_path, recursive=True)
        
        if not files:
            search_path = os.path.join(self.dataset_path, "data", "*.parquet")
            files = glob.glob(search_path, recursive=True)

        if not files and os.path.isfile(self.dataset_path) and self.dataset_path.endswith('.parquet'):
            files = [self.dataset_path]

        if not files:
            raise FileNotFoundError(f"No parquet files found in {self.dataset_path}")
            
        print(f"Loading data from {len(files)} files...")
        dfs = [pd.read_parquet(f) for f in files]
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.df)} examples.")
        
        if self.k_shots > 0:
            print(f"Initializing ShotSelector (strategy={self.selector_strategy}, k={self.k_shots})...")
            self.selector = ShotSelector(
                self.df, 
                method=self.selector_strategy, 
                k=self.k_shots,
                dpo_model_path=self.dpo_model_path
            )

    def predict(self, problem):
        """Sends the problem to Ollama for classification."""
        
        examples_str = ""
        if self.k_shots > 0 and self.selector:
            try:
                examples = self.selector.select(problem, k=self.k_shots)
                examples_str = "Here are some examples:\n\n"
                for i, ex in enumerate(examples):
                    if ex['problem'].strip() == problem.strip():
                        continue
                    label = ex.get('type', ex.get('category', 'Unknown'))
                    examples_str += f"Example {i+1}:\nProblem: {ex['problem']}\nCategory: {label}\n\n"
            except Exception as e:
                print(f"Selector error: {e}")
        
        prompt = f"""
        Classify the following math problem into exactly one of these categories: {', '.join(self.categories)}.
        Return ONLY the category name. Do not include any other text.
        
        {examples_str if examples_str else ""} Problem: {problem}
        
        Category:
        """
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0
            }
        }
        
        try:
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            data = response.json()
            result = data.get("response", "").strip()
            
            # Extract token counts
            prompt_eval_count = data.get("prompt_eval_count", 0)
            eval_count = data.get("eval_count", 0)
            
            # Flexible matching for category
            for cat in sorted(self.categories, key=len, reverse=True):
                if cat.lower() in result.lower():
                    return cat, prompt_eval_count, eval_count
            return "Unknown", prompt_eval_count, eval_count
        except requests.exceptions.RequestException as e:
            print(f"Error querying model: {e}")
            return "Error", 0, 0

    def evaluate(self, sample_size=None, batch_size=1):
        """Runs the pipeline on the dataset and computes accuracy.
        
        Args:
            sample_size: Number of samples to evaluate (None for all)
            batch_size: Number of samples to process before showing progress (default=1)
        """
        if sample_size:
            data = self.df.sample(n=sample_size, random_state=42)
        else:
            data = self.df

        correct = 0
        total = 0
        
        y_true = []
        y_pred = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency = 0.0
        
        print("Starting classification...")
        batch_count = 0
        for idx, (i, row) in enumerate(data.iterrows()):
            problem = row['problem']
            true_label = row.get('type', row.get('category', 'Unknown')) # robust column access
            
            # Unpack prediction and tokens
            import time
            start_time = time.time()
            predicted_label, prompt_tokens, completion_tokens = self.predict(problem)
            latency = time.time() - start_time
            
            y_true.append(true_label)
            y_pred.append(predicted_label)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_latency += latency
            
            is_correct = predicted_label == true_label
            if is_correct:
                correct += 1
            total += 1
            
            # Show progress per batch
            if (idx + 1) % batch_size == 0:
                batch_count += 1
                batch_accuracy = correct / total if total > 0 else 0
                print(f"Batch {batch_count} ({idx + 1}/{len(data)}): Accuracy={batch_accuracy:.2%}, Avg Latency={total_latency/total:.2f}s")
            else:
                print(f"[{i}] True: {true_label} | Pred: {predicted_label} | Correct: {is_correct}")

        accuracy = correct / total if total > 0 else 0
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        avg_prompt_tokens = total_prompt_tokens / total if total > 0 else 0
        avg_completion_tokens = total_completion_tokens / total if total > 0 else 0
        avg_latency = total_latency / total if total > 0 else 0.0
        
        print(f"Accuracy: {accuracy:.2%}")
        print(f"F1 (Weighted): {f1_weighted:.4f}")
        print(f"F1 (Macro): {f1_macro:.4f}")
        print(f"Avg Prompt Tokens: {avg_prompt_tokens:.1f}")
        print(f"Avg Completion Tokens: {avg_completion_tokens:.1f}")
        print(f"Avg Latency: {avg_latency:.4f}s")
        
        return accuracy, f1_weighted, f1_macro, avg_prompt_tokens, avg_completion_tokens, avg_latency

    def run(self, sample_size=5):
        self.load_data()
        self.evaluate(sample_size=sample_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SLM on Math Classification")
    parser.add_argument("--model", type=str, default="llama3:8b", help="Ollama model name")
    parser.add_argument("--dataset", type=str, default="datasets/competition_math/data/", help="Path to dataset")
    parser.add_argument("--sample", type=int, default=5, help="Number of samples to evaluate")
    
    # Added arguments to support standalone testing of strategies
    parser.add_argument("--strategy", type=str, default="random", 
                        choices=["random", "lexical", "semantic", "cross_encoder", "dpo"],
                        help="Shot selection strategy")
    parser.add_argument("--k", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--dpo_path", type=str, default="dpo_selector_model", 
                        help="Path to trained DPO model")
    
    args = parser.parse_args()
    
    pipeline = MetricsPipeline(
        model_name=args.model, 
        dataset_path=args.dataset,
        selector_strategy=args.strategy,
        k_shots=args.k,
        dpo_model_path=args.dpo_path
    )
    pipeline.run(sample_size=args.sample)