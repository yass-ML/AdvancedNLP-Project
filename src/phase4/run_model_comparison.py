import sys
import os
import yaml
import argparse
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from benchmark_models import benchmark_models

def run_comparison(sample_size, k_shots, dataset_path, train_path, dpo_path):
    MODELS = ['llama3:8b', 'mistral:7b', 'gemma:7b', 'phi3:mini', 'qwen2:7b', 'qwen3:8b']
    STRATEGIES = ['semantic', 'dpo']

    all_results = []

    print(f"Starting Model Comparison Experiment")
    print(f"Models: {MODELS}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Sample Size: {sample_size} | K: {k_shots}")

    for strategy in STRATEGIES:
        print(f"\n{'#'*60}")
        print(f"Running for Strategy: {strategy.upper()}")
        print(f"{'#'*60}")

        try:
            results = benchmark_models(
                models=MODELS,
                sample_size=sample_size,
                dataset_path=dataset_path,
                selector_strategy=strategy,
                k_shots=k_shots,
                dpo_path=dpo_path,
                output_dir="experiment_results/classification/4_model-experiment-results",
                train_dataset_path=train_path
            )
            all_results.extend(results)
        except Exception as e:
            print(f"Error running strategy {strategy}: {e}")

    output_dir = "experiment_results/classification/4_model-experiment-results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "model_comparison_results.yaml")

    with open(output_file, 'w') as f:
        yaml.dump(all_results, f, sort_keys=False)

    print(f"\n--> All experiments completed. Aggregated results saved to {output_file}")

    df = pd.DataFrame(all_results)
    if not df.empty:
        print("\nAggregated Summary:")
        print(df[["Model", "Strategy", "Accuracy", "Avg Latency"] if "Avg Latency" in df.columns else ["Model", "Strategy", "Accuracy"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Model Comparison Experiment")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of samples per evaluation")
    parser.add_argument("--k", type=int, default=3, help="Number of shots")
    parser.add_argument("--dataset", type=str, default="datasets/competition_math/data/test.parquet", help="Path to Evaluation Data (Test)")
    parser.add_argument("--train_dataset", type=str, default="datasets/competition_math/data/train.parquet", help="Path to Selection Data (Train)")
    parser.add_argument("--dpo_path", type=str, default="dpo_selector_model")

    args = parser.parse_args()

    run_comparison(args.sample_size, args.k, args.dataset, args.train_dataset, args.dpo_path)
