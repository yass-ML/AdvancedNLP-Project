
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_phase5_results():
    # Load results
    results_path = os.path.join(os.path.dirname(__file__), "../../model-experiment-result/phase5_scaling_results.yaml")
    
    if not os.path.exists(results_path):
        print(f"Error: Results file not found at {results_path}")
        return

    with open(results_path, 'r') as f:
        data = yaml.safe_load(f)
    
    df = pd.DataFrame(data)
    
    # Filter for success only
    df = df[df['Status'] == 'Success']
    
    # Sort by K to ensure line plots are correct
    df = df.sort_values(by='K')
    
    # Setup styling
    sns.set_theme(style="whitegrid")
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Phase 5: K-Shot Scaling Experiment Results (Llama-3-8B + DPO)', fontsize=16)
    
    # Plot 1: Accuracy vs K
    sns.lineplot(ax=axes[0], data=df, x='K', y='Accuracy', marker='o', linewidth=2.5, color='b')
    axes[0].set_title('Accuracy vs Shot Count (K)', fontsize=14)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_xlabel('K (Number of Shots)', fontsize=12)
    axes[0].set_ylim(0, 1.0)
    
    # Annotate max accuracy
    max_acc = df['Accuracy'].max()
    max_k = df.loc[df['Accuracy'] == max_acc, 'K'].iloc[0]
    axes[0].annotate(f'Peak: {max_acc:.2f} (K={max_k})', 
                     xy=(max_k, max_acc), 
                     xytext=(max_k, max_acc + 0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=10, ha='center')

    # Plot 2: Latency vs K
    sns.lineplot(ax=axes[1], data=df, x='K', y='Avg_Latency', marker='s', linewidth=2.5, color='r')
    axes[1].set_title('Average Latency vs K', fontsize=14)
    axes[1].set_ylabel('Latency (s)', fontsize=12)
    axes[1].set_xlabel('K (Number of Shots)', fontsize=12)
    
    # Plot 3: Prompt Tokens vs K
    sns.lineplot(ax=axes[2], data=df, x='K', y='Avg_Prompt_Tokens', marker='^', linewidth=2.5, color='g')
    axes[2].set_title('Prompt Token Usage vs K', fontsize=14)
    axes[2].set_ylabel('Avg Prompt Tokens', fontsize=12)
    axes[2].set_xlabel('K (Number of Shots)', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), "../../graphs")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "phase5_scaling_analysis.png")
    
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    plot_phase5_results()
