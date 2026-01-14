import argparse
import yaml
from ner_task_metrics import NERMetricsPipeline

def run_zeroshot(output_file="ner_task_zeroshot_results.yaml", sample_size=50, debug_mode=False):
    results = {}
    models = ["llama3:8b", "phi3:mini", "qwen3:8b"]

    print(f"\n=== Running Zero-Shot (K=0) Evaluation ===")

    zeroshot_results = []

    for model in models:
        print(f"Testing Model: {model} (Zero-Shot)")

        # Train path unused for K=0 but passed for consistency
        train_path = "datasets/few_nerd/train.parquet"

        pipeline = NERMetricsPipeline(
            model_name=model,
            k_shots=0,
            selector_strategy="random", # Irrelevant for K=0
            dpo_model_path="dpo_selector_model_ner", # Irrelevant
            train_dataset_path=train_path,
            debug=debug_mode
        )

        # Loads test data. Skips train data loading since K=0
        pipeline.load_data()
        res = pipeline.evaluate(sample_size=sample_size)

        # Enhance result info
        res['model'] = model
        res['k'] = 0
        res['strategy'] = 'zero_shot'

        zeroshot_results.append(res)

    results['zeroshot_evaluation'] = zeroshot_results

    # Save to YAML
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        yaml.dump(results, f)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run with tiny sample size")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--sample", type=int, default=0, help="Override sample size (0 for default)")
    args = parser.parse_args()

    if args.sample > 0:
        sample = args.sample
    else:
        sample = 10 if args.test else 50

    run_zeroshot(sample_size=sample, debug_mode=args.debug)
