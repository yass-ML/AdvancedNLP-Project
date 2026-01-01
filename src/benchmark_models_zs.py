import argparse
import pandas as pd
from compute_metrics import MetricsPipeline

import subprocess

def ensure_model_exists(model_name):
    """Checks if model exists in Ollama, pulls if not."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        if model_name not in result.stdout:
            print(f"Model {model_name} not found locally. Pulling...")
            subprocess.run(["ollama", "pull", model_name], check=True)
            print(f"Successfully pulled {model_name}.")
        else:
            print(f"Model {model_name} found.")
    except subprocess.CalledProcessError as e:
        print(f"Error checking/pulling model {model_name}: {e}")
        raise

def benchmark_models(models, sample_size, dataset_path):
    results = []
    
    print(f"Starting benchmark on {len(models)} models with sample_size={sample_size}...")
    
    for model in models:
        print(f"\n{'='*50}")
        print(f"Evaluating model: {model}")
        print(f"{'='*50}")
        
        try:
            ensure_model_exists(model)
            
            pipeline = MetricsPipeline(model_name=model, dataset_path=dataset_path)
            pipeline.load_data()
            accuracy, f1_weighted, f1_macro, avg_prompt_tokens, avg_completion_tokens = pipeline.evaluate(sample_size=sample_size)
            
            results.append({
                "Model": model,
                "Accuracy": accuracy,
                "F1 (Weighted)": f1_weighted,
                "F1 (Macro)": f1_macro,
                "Avg Prompt Tokens": avg_prompt_tokens,
                "Avg Completion Tokens": avg_completion_tokens,
                "Status": "Success"
            })
            print(f"Model {model} matched with accuracy: {accuracy:.2%}")
            
        except Exception as e:
            print(f"Failed to evaluate {model}: {e}")
            results.append({
                "Model": model,
                "Accuracy": 0.0,
                "F1 (Weighted)": 0.0,
                "F1 (Macro)": 0.0,
                "Avg Prompt Tokens": 0.0,
                "Avg Completion Tokens": 0.0,
                "Status": f"Failed: {str(e)}"
            })
            
    df_results = pd.DataFrame(results)
    output_file = "benchmark_results.csv"
    df_results.to_csv(output_file, index=False)
    
    print(f"\nBenchmark completed. Results saved to {output_file}")
    print(df_results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SLMs on Math Classification")
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["llama3:8b", "mistral:7b", "gemma:7b", "phi3:mini", "qwen2:7b"],
                        help="List of Ollama models to benchmark")
    parser.add_argument("--dataset", type=str, default="datasets/competition_math/data/", help="Path to dataset")
    parser.add_argument("--sample", type=int, default=5, help="Number of samples to evaluate per model")
    
    args = parser.parse_args()
    
    benchmark_models(args.models, args.sample, args.dataset)
