
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_baseline():
    # Path to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "baseline_benchmark_results.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Sort by Accuracy
    df = df.sort_values(by="Accuracy", ascending=False)
    
    # Setup plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Bar plot for Accuracy
    ax = sns.barplot(data=df, x="Model", y="Accuracy", palette="viridis")
    
    # Add values on top of bars
    for i in ax.containers:
        ax.bar_label(i, fmt='%.3f', padding=3)

    plt.title("Zero-Shot Baseline Accuracy (No Examples)", fontsize=16)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.ylim(0, 1.0)
    
    # Save
    output_dir = os.path.join(os.path.dirname(__file__), "../../graphs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "zero_shot_baseline_accuracy.png")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.close()
    
    # Secondary Plot: Accuracy vs tokens (scatter?) or just F1?
    # Let's do a grouped bar for Accuracy vs F1 Weighted
    
    df_melted = df.melt(id_vars=["Model"], value_vars=["Accuracy", "F1 (Weighted)"], var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="muted")
    
    for i in ax.containers:
        ax.bar_label(i, fmt='%.2f', padding=3)
        
    plt.title("Zero-Shot Baseline: Accuracy vs F1-Weighted", fontsize=16)
    plt.ylim(0, 1.0)
    
    output_file_comp = os.path.join(output_dir, "zero_shot_baseline_comparison.png")
    plt.tight_layout()
    plt.savefig(output_file_comp, dpi=300)
    print(f"Comparison plot saved to {output_file_comp}")
    plt.close()

if __name__ == "__main__":
    plot_baseline()
