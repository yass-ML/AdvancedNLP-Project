
import sys
import os
import yaml
import argparse
import pandas as pd

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compute_metrics import MetricsPipeline

def resolve_path(provided_path, target_name):
    """
    Helper to resolve paths relative to CWD or Project Root.
    """
    if os.path.exists(provided_path):
        return provided_path
    
    # Try finding it in project root (assuming script is in src/phase5/ -> root is ../../)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    candidates = [
        os.path.join(project_root, target_name),
        os.path.join(project_root, "src", target_name),
        os.path.join(os.getcwd(), target_name)
    ]
    
    for p in candidates:
        if os.path.exists(p):
            return p
            
    return provided_path # Return original to fail with clear error if not found

def run_scaling_experiment(sample_size, dpo_path_arg, batch_size=10):
    # Fixed Parameters
    MODELS = ['deepseek-r1:8b']
    STRATEGY = 'dpo'
    
    # Resolve Paths
    dataset_relative = "datasets/competition_math/data/train-00000-of-00001-7320a6f3aba8ebd2.parquet"
    DATASET_PATH = resolve_path(dataset_relative, dataset_relative)
    
    # Clean up dpo path argument if it was the default relative one which failed
    DPO_PATH = resolve_path(dpo_path_arg, "dpo_selector_model")
    
    # Variable: K-Shots
    K_VALUES = [1, 3, 5, 10, 15, 20, 25]
    
    all_results = []

    for model in MODELS:
        print(f"\n{'#'*60}")
        print(f"Running for Model: {model}")
        print(f"{'#'*60}")
    
        print(f"Starting Phase 5: K-Shot Scaling Experiment")
        print(f"Model: {model} | Strategy: {STRATEGY}")
        print(f"Search Paths - Dataset: {DATASET_PATH}")
        print(f"Search Paths - DPO Model: {DPO_PATH}")
        print(f"K Values: {K_VALUES}")
        print(f"Sample Size: {sample_size}")
        print(f"Batch Size: {batch_size}")
    
        if not os.path.exists(DPO_PATH):
            print(f"CRITICAL ERROR: DPO Model path not found: {DPO_PATH}")
            sys.exit(1)

        for k in K_VALUES:
            print(f"\n{'#'*60}")
            print(f"Running for K = {k}")
            print(f"{'#'*60}")
        
            try:
                pipeline = MetricsPipeline(
                    model_name=model,
                    dataset_path=DATASET_PATH,
                    selector_strategy=STRATEGY,
                    k_shots=k,
                    dpo_model_path=DPO_PATH
                )
                
                pipeline.load_data()
                
                # Evaluate returns: accuracy, f1_weighted, f1_macro, avg_prompt_tokens, avg_completion_tokens, avg_latency
                print(f"    Processing {sample_size} samples in batches of {batch_size}...")
                accuracy, f1_w, f1_m, avg_prompt, avg_comp, avg_latency = pipeline.evaluate(
                    sample_size=sample_size,
                    batch_size=batch_size
                )
                
                result_entry = {
                    "K": k,
                    "Accuracy": float(accuracy),
                    "F1_Weighted": float(f1_w),
                    "F1_Macro": float(f1_m),
                    "Avg_Prompt_Tokens": float(avg_prompt),
                    "Avg_Completion_Tokens": float(avg_comp),
                    "Avg_Latency": float(avg_latency),
                    "Model": model,
                    "Strategy": STRATEGY,
                    "Status": "Success"
                }
                
                all_results.append(result_entry)
                print(f"--> K={k} Result: Accuracy={accuracy:.2%}, Latency={avg_latency:.2f}s")
                
            except Exception as e:
                print(f"Error running K={k}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "K": k,
                    "Status": f"Failed: {str(e)}",
                    "Model": model,
                    "Strategy": STRATEGY
                })

    output_dir = os.path.join(os.path.dirname(__file__), "../../model-experiment-result")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "phase5_scaling_results.yaml")
    
    with open(output_file, 'w') as f:
        yaml.dump(all_results, f, sort_keys=False)
        
    print(f"\n--> Experiment completed. Results saved to {output_file}")
    
    # Simple display
    df = pd.DataFrame(all_results)
    if not df.empty:
        print("\nAggregated Summary:")
        cols = ["K", "Accuracy", "Avg_Prompt_Tokens", "Avg_Latency"] if "Avg_Latency" in df.columns else ["K", "Status"]
        available_cols = [c for c in cols if c in df.columns]
        print(df[available_cols])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phase 5 Scaling Experiment")
    parser.add_argument("--sample_size", type=int, default=100, help="Number of samples per K")
    parser.add_argument("--dpo_path", type=str, default="dpo_selector_model", help="Path to DPO model")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for processing samples")
    
    args = parser.parse_args()
    
    run_scaling_experiment(args.sample_size, args.dpo_path, args.batch_size)
