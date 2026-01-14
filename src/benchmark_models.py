import argparse
import pandas as pd
from compute_metrics import MetricsPipeline
import yaml
import subprocess
import os

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

def benchmark_models(models, sample_size, dataset_path, selector_strategy, k_shots, dpo_path, output_dir="few_shot_results", train_dataset_path=None):
    results = []

    print(f"Starting benchmark...")
    print(f"Strategy: {selector_strategy} | K: {k_shots} | DPO Path: {dpo_path}")

    for model in models:
        print(f"\n{'='*50}")
        print(f"Evaluating model: {model}")
        print(f"{'='*50}")

        try:
            ensure_model_exists(model)

            pipeline = MetricsPipeline(
                model_name=model,
                dataset_path=dataset_path,
                selector_strategy=selector_strategy,
                k_shots=k_shots,
                dpo_model_path=dpo_path,
                train_dataset_path=train_dataset_path
            )

            pipeline.load_data()

            accuracy, f1_w, f1_m, avg_prompt, avg_comp, avg_latency = pipeline.evaluate(sample_size=sample_size)

            results.append({
                "Model": model,
                "Accuracy": accuracy,
                "F1 (Weighted)": f1_w,
                "F1 (Macro)": f1_m,
                "Avg Prompt Tokens": avg_prompt,
                "Avg Completion Tokens": avg_comp,
                "Avg Latency": avg_latency,
                "Strategy": selector_strategy,
                "K": k_shots,
                "Status": "Success"
            })
            print(f"Model {model} Result: Accuracy={accuracy:.2%}")

        except Exception as e:
            print(f"Failed to evaluate {model}: {e}")
            results.append({
                "Model": model,
                "Accuracy": 0.0,
                "Strategy": selector_strategy,
                "Status": f"Failed: {str(e)}"
            })

    df_results = pd.DataFrame(results)

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"benchmark_{selector_strategy}_k{k_shots}.yaml"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, "w+") as file:
        yaml.dump(df_results.to_dict(orient='records'), file)

    print(f"\nBenchmark completed. Results saved to {output_path}")
    print(df_results[["Model", "Accuracy", "Strategy"]])

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark SLMs on Math Classification")
    parser.add_argument("--models", type=str, nargs="+",
                        default=["phi3:mini", "llama3:8b"],
                        help="List of Ollama models")
    parser.add_argument("--dataset", type=str, default="datasets/competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet")
    parser.add_argument("--sample", type=int, default=10, help="Number of samples")

    parser.add_argument("--strategy", type=str, default="random",
                        choices=["random", "lexical", "semantic", "cross_encoder", "dpo"],
                        help="Shot selection strategy")
    parser.add_argument("--k", type=int, default=3, help="Number of few-shot examples")

    parser.add_argument("--dpo_path", type=str, default="dpo_selector_model",
                        help="Path to the trained DPO Cross-Encoder model")

    args = parser.parse_args()

    benchmark_models(args.models, args.sample, args.dataset, args.strategy, args.k, args.dpo_path)
