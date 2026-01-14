import argparse
import yaml
import os
from ner_task_metrics import NERMetricsPipeline

def run_scaling_only(output_file="ner_task_scaling_results.yaml", sample_size=50, debug_mode=False):
    results = {}

    # Configuration
    strategies = ["dpo"] # Using DPO as the default best strategy
    models = ["llama3:8b", "phi3:mini", "qwen3:8b"]
    k_values = [1, 3, 5, 10, 15, 20, 25]

    print(f"\n=== Running Scaling Experiment ONLY ===")
    print(f"Models: {models}")
    print(f"K Values: {k_values}")
    print(f"Strategy: {strategies[0]}")

    scaling_results = []

    # Loop over all models
    for model in models:
        for k in k_values:
            print(f"Testing Model: {model}, K={k}")

            # Use small dataset for tests if sample size is small to verify pipeline quickly
            train_path = "datasets/few_nerd/train_small.parquet" if sample_size < 50 else "datasets/few_nerd/train.parquet"

            pipeline = NERMetricsPipeline(
                model_name=model,
                k_shots=k,
                selector_strategy=strategies[0],
                dpo_model_path="dpo_selector_model_ner",
                train_dataset_path=train_path,
                debug=debug_mode
            )

            try:
                pipeline.load_data()
                res = pipeline.evaluate(sample_size=sample_size)
                # Ensure model name is explicitly correct in result if not already
                res['model'] = model
                scaling_results.append(res)
            except Exception as e:
                print(f"Error evaluating {model} at K={k}: {e}")

    results['scaling_experiment'] = scaling_results

    # Save to YAML
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        yaml.dump(results, f)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run with tiny sample size")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--sample", type=int, default=0, help="Override sample size")
    args = parser.parse_args()

    if args.sample > 0:
        sample = args.sample
    else:
        sample = 10 if args.test else 50

    run_scaling_only(sample_size=sample, debug_mode=args.debug)
