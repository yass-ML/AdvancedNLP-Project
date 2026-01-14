import argparse
import yaml
import os
import sys

# Add current directory to path just in case, though usually implicit for script dir
# But if run as module...
# Actually, if both are in src/ner, 'from ner_task_metrics' works fine if run from src/ner.
# But let's be safe.
from ner_task_metrics import NERMetricsPipeline

def run_experiments(output_file="ner_task_results.yaml", sample_size=50, debug_mode=False):
    results = {}

    # Common settings
    k_default = 3
    model_default = "llama3:8b"
    strategies = ["random", "lexical", "semantic", "dpo"]
    models = ["llama3:8b", "phi3:mini", "qwen3:8b"]
    k_values = [1, 3, 5, 10, 15, 20, 25]

    # 1. Selector Experiment
    print("\n=== Running Selector Experiment ===")
    selector_results = []
    # Use small dataset for tests to speed up encoding
    train_path = "datasets/few_nerd/train_small.parquet" if sample_size < 50 else "datasets/few_nerd/train.parquet"

    for strategy in strategies:
        print(f"Testing Strategy: {strategy}")
        pipeline = NERMetricsPipeline(
            model_name=model_default,
            k_shots=k_default,
            selector_strategy=strategy,
            dpo_model_path="dpo_selector_model_ner",
            train_dataset_path=train_path,
            debug=debug_mode
        )
        pipeline.load_data()
        res = pipeline.evaluate(sample_size=sample_size)
        selector_results.append(res)
    results['selector_experiment'] = selector_results

    # Find best strategy
    best_strategy = "dpo" # Default assumption
    best_f1 = -1
    for r in selector_results:
        if r['f1'] > best_f1:
            best_f1 = r['f1']
            best_strategy = r['strategy']
    print(f"Best Strategy identified: {best_strategy}")

    # 2. Model Experiment
    print("\n=== Running Model Experiment ===")
    model_results = []
    for model in models:
        print(f"Testing Model: {model}")
        pipeline = NERMetricsPipeline(
            model_name=model,
            k_shots=k_default,
            selector_strategy=best_strategy,
            dpo_model_path="dpo_selector_model_ner",
            train_dataset_path=train_path,
            debug=debug_mode
        )
        # Re-load data? Optimization: load once if class structure allows, but safe to reload.
        pipeline.load_data()
        res = pipeline.evaluate(sample_size=sample_size)
        model_results.append(res)
    results['model_experiment'] = model_results

    # Find best model
    best_model = "llama3:8b"
    best_model_f1 = -1
    for r in model_results:
        if r['f1'] > best_model_f1:
            best_model_f1 = r['f1']
            best_model = r['model']
    print(f"Best Model identified: {best_model}")

    # 3. Scaling Experiment
    print("\n=== Running Scaling Experiment ===")
    scaling_results = []
    for k in k_values:
        print(f"Testing K={k} with {best_model} and {best_strategy}")
        pipeline = NERMetricsPipeline(
            model_name=best_model,
            k_shots=k,
            selector_strategy=best_strategy,
            dpo_model_path="dpo_selector_model_ner",
            train_dataset_path=train_path,
            debug=debug_mode
        )
        pipeline.load_data()
        res = pipeline.evaluate(sample_size=sample_size)
        scaling_results.append(res)
    results['scaling_experiment'] = scaling_results

    # Find best K
    best_k = 3
    best_k_f1 = -1
    for r in scaling_results:
        if r['f1'] > best_k_f1:
            best_k_f1 = r['f1']
            best_k = r['k']
    print(f"Best K identified: {best_k}")

    # 4. Final Evaluation (Best Model/Selector/K on ALL models)
    print("\n=== Running Final Evaluation ===")
    print(f"Using Strategy={best_strategy}, K={best_k} for ALL models.")
    final_results = []
    for model in models:
        print(f"Final Eval: {model}")
        pipeline = NERMetricsPipeline(
            model_name=model,
            k_shots=best_k,
            selector_strategy=best_strategy,
            dpo_model_path="dpo_selector_model_ner",
            train_dataset_path=train_path,
            debug=debug_mode
        )
        # Optimization: load_data called inside evaluate if integrated? No, explicit load.
        pipeline.load_data()
        res = pipeline.evaluate(sample_size=sample_size)
        final_results.append(res)
    results['final_evaluation'] = final_results

    # Save to YAML
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        yaml.dump(results, f)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run with tiny sample size")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    sample = 10 if args.test else 50 # Or full? User asked for 'full pipeline' implies running on full test set maybe?
    # But let's stick to sample size arg logic. If user wants full, they'd change the code or we provide arg.
    # User said "run experiment" not "evaluate all".
    # But for a proper run, 50 is small. Let's make it bigger if not test?
    # Actually, let's keep it configurable or default 50 for now, as full run is very slow (2s/it * 1000s = hours).

    run_experiments(sample_size=sample, debug_mode=args.debug)
