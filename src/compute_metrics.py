
import pandas as pd
import requests
import argparse
import os
import glob

class MetricsPipeline:
    def __init__(self, model_name="llama3:8b", dataset_path="datasets/competition_math"):
        self.model_name = model_name
        self.dataset_path = dataset_path
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

    def predict(self, problem):
        """Sends the problem to Ollama for classification."""
        prompt = f"""
        Classify the following math problem into exactly one of these categories: {', '.join(self.categories)}.
        Return ONLY the category name. Do not include any other text.
        
        Problem: {problem}
        
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
            result = response.json().get("response", "").strip()
            
            for cat in self.categories:
                if cat.lower() in result.lower():
                    return cat
            return "Unknown"
        except requests.exceptions.RequestException as e:
            print(f"Error querying model: {e}")
            return "Error"

    def evaluate(self, sample_size=None):
        """Runs the pipeline on the dataset and computes accuracy."""
        if sample_size:
            data = self.df.sample(n=sample_size, random_state=42)
        else:
            data = self.df

        correct = 0
        total = 0
        
        print("Starting classification...")
        for i, row in data.iterrows():
            problem = row['problem']
            true_label = row['type']

            predicted_label = self.predict(problem)
            
            is_correct = predicted_label == true_label
            if is_correct:
                correct += 1
            total += 1
            
            print(f"[{i}] True: {true_label} | Pred: {predicted_label} | Correct: {is_correct}")

        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy:.2%}")
        return accuracy

    def run(self, sample_size=5):
        self.load_data()
        self.evaluate(sample_size=sample_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SLM on Math Classification")
    parser.add_argument("--model", type=str, default="llama3:8b", help="Ollama model name")
    parser.add_argument("--dataset", type=str, default="../datasets/competition_math", help="Path to dataset")
    parser.add_argument("--sample", type=int, default=5, help="Number of samples to evaluate")
    
    args = parser.parse_args()
    
    pipeline = MetricsPipeline(model_name=args.model, dataset_path=args.dataset)
    pipeline.run(sample_size=args.sample)