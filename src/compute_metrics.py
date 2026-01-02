
import pandas as pd
from sklearn.metrics import f1_score
import requests
import argparse
import os
import glob
from shot_selector import ShotSelector

class MetricsPipeline:
    def __init__(self, model_name="llama3:8b", dataset_path="datasets/competition_math", selector_strategy="random", k_shots=0):
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.selector_strategy = selector_strategy
        self.k_shots = k_shots
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

        if not files:
            raise FileNotFoundError(f"No parquet files found in {self.dataset_path}")
            
        print(f"Loading data from {len(files)} files...")
        dfs = [pd.read_parquet(f) for f in files]
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(self.df)} examples.")
        
        if self.k_shots > 0:
            print(f"Initializing ShotSelector (strategy={self.selector_strategy}, k={self.k_shots})...")
            self.selector = ShotSelector(self.df, method=self.selector_strategy, k=self.k_shots)

    def predict(self, problem):
        """Sends the problem to Ollama for classification."""
        
        examples_str = ""
        try:
            examples = self.selector.select(problem, k=self.k_shots)
            examples_str = "Here are some examples:\n\n"
            for i, ex in enumerate(examples):
                # Basic filtering: don't use the problem itself as an example
                if ex['problem'].strip() == problem.strip():
                    continue
                examples_str += f"Example {i+1}:\nProblem: {ex['problem']}\nCategory: {ex['type']}\n\n"
        except Exception as e:
            print(f"Selector error: {e}")
        
        prompt = f"""
        Classify the following math problem into exactly one of these categories: {', '.join(self.categories)}.
        Return ONLY the category name. Do not include any other text.
        
        {examples_str}Problem: {problem}
        
        Category:
        """
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        try:
            response = requests.post(self.base_url, json=payload)
            response.raise_for_status()
            data = response.json()
            result = data.get("response", "").strip()
            
            # Extract token counts
            prompt_eval_count = data.get("prompt_eval_count", 0)
            eval_count = data.get("eval_count", 0)
            
            for cat in sorted(self.categories, key=len, reverse=True):
                if cat.lower() in result.lower():
                    return cat, prompt_eval_count, eval_count
            return "Unknown", prompt_eval_count, eval_count
        except requests.exceptions.RequestException as e:
            print(f"Error querying model: {e}")
            return "Error", 0, 0

    def evaluate(self, sample_size=None):
        """Runs the pipeline on the dataset and computes accuracy."""
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
        
        print("Starting classification...")
        for i, row in data.iterrows():
            problem = row['problem']
            true_label = row['type']
            
            # Unpack prediction and tokens
            predicted_label, prompt_tokens, completion_tokens = self.predict(problem)
            
            y_true.append(true_label)
            y_pred.append(predicted_label)
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            
            is_correct = predicted_label == true_label
            if is_correct:
                correct += 1
            total += 1
            
            print(f"[{i}] True: {true_label} | Pred: {predicted_label} | Correct: {is_correct}")

        accuracy = correct / total if total > 0 else 0
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        avg_prompt_tokens = total_prompt_tokens / total if total > 0 else 0
        avg_completion_tokens = total_completion_tokens / total if total > 0 else 0
        
        print(f"Accuracy: {accuracy:.2%}")
        print(f"F1 (Weighted): {f1_weighted:.4f}")
        print(f"F1 (Macro): {f1_macro:.4f}")
        print(f"Avg Prompt Tokens: {avg_prompt_tokens:.1f}")
        print(f"Avg Completion Tokens: {avg_completion_tokens:.1f}")
        
        return accuracy, f1_weighted, f1_macro, avg_prompt_tokens, avg_completion_tokens

    def run(self, sample_size=5):
        self.load_data()
        self.evaluate(sample_size=sample_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SLM on Math Classification")
    parser.add_argument("--model", type=str, default="llama3:8b", help="Ollama model name")
    parser.add_argument("--dataset", type=str, default="datasets/competition_math/data/", help="Path to dataset")
    parser.add_argument("--sample", type=int, default=5, help="Number of samples to evaluate")
    
    args = parser.parse_args()
    
    pipeline = MetricsPipeline(model_name=args.model, dataset_path=args.dataset)
    pipeline.run(sample_size=args.sample)