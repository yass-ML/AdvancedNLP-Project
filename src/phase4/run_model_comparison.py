import sys
import os
import yaml
import argparse
import pandas as pd

# Add src to python path to allow importing benchmark_models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmark_models import benchmark_models

def run_comparison(sample_size, k_shots, dataset_path, dpo_path):
    MODELS = ['llama3:8b', 'mistral:7b', 'gemma:7b', 'phi3:mini']
    STRATEGIES = ['semantic']
    
    all_results = []
    
    print(f"Starting Model Comparison Experiment")
    print(f"Models: {MODELS}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Sample Size: {sample_size} | K: {k_shots}")
    
    for strategy in STRATEGIES:
        print(f"\n{'#'*60}")
        print(f"Running for Strategy: {strategy.upper()}")
        print(f"{'#'*60}")
        
        # We run benchmark_models for all models for this strategy
        # benchmark_models handles the loop over models, but we can also loop manually if we want more control.
        # However, benchmark_models takes a list of models, so let's use that feature.
        
        try:
            results = benchmark_models(
                models=MODELS,
                sample_size=sample_size,
                dataset_path=dataset_path,
                selector_strategy=strategy,
                k_shots=k_shots,
                dpo_path=dpo_path,
                output_dir="model-experiment-result"
            )
            all_results.extend(results)
        except Exception as e:
            print(f"Error running strategy {strategy}: {e}")
            
    # Save aggregated results
    output_dir = "model-experiment-result"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "model_comparison_results.yaml")
    
    with open(output_file, 'w') as f:
        yaml.dump(all_results, f, sort_keys=False)
        
    print(f"\n--> All experiments completed. Aggregated results saved to {output_file}")
    
    # Simple display
    df = pd.DataFrame(all_results)
    if not df.empty:
        print("\nAggregated Summary:")
        print(df[["Model", "Strategy", "Accuracy", "Avg Latency"] if "Avg Latency" in df.columns else ["Model", "Strategy", "Accuracy"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Model Comparison Experiment")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of samples per evaluation")
    parser.add_argument("--k", type=int, default=3, help="Number of shots")
    parser.add_argument("--dataset", type=str, default="../datasets/competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet")
    parser.add_argument("--dpo_path", type=str, default="../dpo_selector_model")
    
    args = parser.parse_args()
    
    run_comparison(args.sample_size, args.k, args.dataset, args.dpo_path)
