"""
Script to run Phase 3 Selector Experiment with various selection strategies.
Saves results in YAML format for further analysis and plotting.

The tested strategies include:
- Random Selection
- Semantic Similarity Selection
- Lexical Similarity Selection
- Cross-Encoder Based Selection
- DPO-Based Selection

"""

import os
import sys
import yaml
import argparse
import subprocess

# Add parent directory to path to find compute_metrics.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compute_metrics import MetricsPipeline

# Configuration
MODEL_NAME = "llama3:8b"
DATASET_PATH = "datasets/competition_math/data/test.parquet"
TRAIN_DATASET_PATH = "datasets/competition_math/data/train.parquet"
DPO_MODEL_PATH = "dpo_selector_model"
STRATEGIES = ["random", "semantic", "lexical", "cross_encoder", "dpo"]
K_SHOTS = 3
SAMPLE_SIZE = 100
OUTPUT_DIR = "experiment_results/classification/3_selector_experiment_results"

def ensure_model_exists(model_name):
    try:
        subprocess.run(["ollama", "list"], capture_output=True, check=True)
    except:
        pass

def run_experiment():
    print(f"Starting Phase 3: Selector Experiment")
    print(f"Model: {MODEL_NAME} | K: {K_SHOTS} | Samples: {SAMPLE_SIZE}")
    print(f"Strategies: {STRATEGIES}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    project_root = os.getcwd()
    if not os.path.exists("datasets"):
        print("Warning: It seems you are not running from project root. Paths might fail.")

    for strategy in STRATEGIES:
        print(f"\n{'='*60}")
        print(f"Running Strategy: {strategy.upper()}")
        print(f"{'='*60}")

        try:
            pipeline = MetricsPipeline(
                model_name=MODEL_NAME,
                dataset_path=DATASET_PATH,
                selector_strategy=strategy,
                k_shots=K_SHOTS,
                dpo_model_path="dpo_selector_model",
                train_dataset_path=TRAIN_DATASET_PATH
            )

            pipeline.load_data()

            accuracy, f1_w, f1_m, avg_prompt, avg_comp, avg_latency = pipeline.evaluate(sample_size=SAMPLE_SIZE)

            result_data = [{
                "Model": MODEL_NAME,
                "Strategy": strategy,
                "K": K_SHOTS,
                "Accuracy": float(accuracy),
                "F1_Weighted": float(f1_w),
                "F1_Macro": float(f1_m),
                "Avg_Prompt_Tokens": float(avg_prompt),
                "Avg_Completion_Tokens": float(avg_comp),
                "Avg_Latency_Seconds": float(avg_latency),
                "Task": "Phase 3 Selector Experiment"
            }]

            filename = f"phase3_selector_experiment_{strategy}.yaml"
            filepath = os.path.join(OUTPUT_DIR, filename)

            with open(filepath, 'w') as f:
                yaml.dump(result_data, f, sort_keys=False)

            print(f"--> Saved results to {filepath}")

        except Exception as e:
            print(f"Error executing strategy {strategy}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    run_experiment()
