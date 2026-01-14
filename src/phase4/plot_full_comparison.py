import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data

def create_dataframe(data):
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(data)

    # Clean up the data - remove entries with Status != Success
    df = df[df['Status'] == 'Success'].copy()

    # Ensure numeric columns are properly typed
    numeric_cols = ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)', 'Avg Prompt Tokens',
                    'Avg Completion Tokens', 'Avg Latency']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def plot_f1_weighted_comparison(df, output_dir):
    """Create a grouped bar chart comparing F1 Weighted scores for Semantic vs DPO"""

    # Filter for semantic and dpo strategies
    df_filtered = df[df['Strategy'].isin(['semantic', 'dpo'])].copy()

    # Sort by model name for consistent ordering
    df_filtered = df_filtered.sort_values('Model')

    plt.figure(figsize=(12, 7))

    # Create the grouped bar plot
    ax = sns.barplot(
        data=df_filtered,
        x='Model',
        y='F1 (Weighted)',
        hue='Strategy',
        palette={'semantic': '#2E86AB', 'dpo': '#A23B72'}
    )

    # Customize the plot
    plt.title('F1 Weighted Score: Semantic vs DPO Strategy Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('F1 Weighted Score', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.ylim(0, 1.0)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3, fontsize=9)

    # Customize legend
    plt.legend(title='Strategy', title_fontsize=11, fontsize=10, loc='upper right')

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_weighted_semantic_vs_dpo.png'), dpi=300)
    plt.close()

    print(f"✓ F1 Weighted comparison plot saved")

def plot_f1_comparison_all_metrics(df, output_dir):
    """Create a comprehensive comparison with multiple subplots"""

    df_filtered = df[df['Strategy'].isin(['semantic', 'dpo'])].copy()
    df_filtered = df_filtered.sort_values('Model')

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Semantic vs DPO Strategy Comparison - All Metrics', fontsize=18, fontweight='bold')

    palette = {'semantic': '#2E86AB', 'dpo': '#A23B72'}

    # F1 Weighted
    sns.barplot(data=df_filtered, x='Model', y='F1 (Weighted)', hue='Strategy',
                ax=axes[0, 0], palette=palette)
    axes[0, 0].set_title('F1 Weighted Score', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('F1 Weighted', fontsize=11)
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].tick_params(axis='x', rotation=45)
    for container in axes[0, 0].containers:
        axes[0, 0].bar_label(container, fmt='%.3f', fontsize=8)

    # Accuracy
    sns.barplot(data=df_filtered, x='Model', y='Accuracy', hue='Strategy',
                ax=axes[0, 1], palette=palette)
    axes[0, 1].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy', fontsize=11)
    axes[0, 1].set_ylim(0, 1.0)
    axes[0, 1].tick_params(axis='x', rotation=45)
    for container in axes[0, 1].containers:
        axes[0, 1].bar_label(container, fmt='%.3f', fontsize=8)

    # F1 Macro
    sns.barplot(data=df_filtered, x='Model', y='F1 (Macro)', hue='Strategy',
                ax=axes[1, 0], palette=palette)
    axes[1, 0].set_title('F1 Macro Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('F1 Macro', fontsize=11)
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].tick_params(axis='x', rotation=45)
    for container in axes[1, 0].containers:
        axes[1, 0].bar_label(container, fmt='%.3f', fontsize=8)

    # Latency - Filter out outliers for better visualization
    df_plot = df_filtered[df_filtered['Avg Latency'] < 10].copy()  # Remove qwen3:8b outlier
    sns.barplot(data=df_plot, x='Model', y='Avg Latency', hue='Strategy',
                ax=axes[1, 1], palette=palette)
    axes[1, 1].set_title('Average Latency (excluding outliers)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Latency (seconds)', fontsize=11)
    axes[1, 1].tick_params(axis='x', rotation=45)
    for container in axes[1, 1].containers:
        axes[1, 1].bar_label(container, fmt='%.2f', fontsize=8)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'semantic_vs_dpo_all_metrics.png'), dpi=300)
    plt.close()

    print(f"✓ All metrics comparison plot saved")

def plot_difference_heatmap(df, output_dir):
    """Create a heatmap showing the difference between DPO and Semantic"""

    # Pivot data to compare strategies
    df_filtered = df[df['Strategy'].isin(['semantic', 'dpo'])].copy()

    metrics = ['Accuracy', 'F1 (Weighted)', 'F1 (Macro)']
    models = sorted(df_filtered['Model'].unique())

    # Create difference matrix (DPO - Semantic)
    diff_matrix = []
    for model in models:
        row = []
        for metric in metrics:
            semantic_val = df_filtered[(df_filtered['Model'] == model) &
                                      (df_filtered['Strategy'] == 'semantic')][metric].values
            dpo_val = df_filtered[(df_filtered['Model'] == model) &
                                 (df_filtered['Strategy'] == 'dpo')][metric].values

            if len(semantic_val) > 0 and len(dpo_val) > 0:
                diff = dpo_val[0] - semantic_val[0]
                row.append(diff)
            else:
                row.append(0)
        diff_matrix.append(row)

    diff_df = pd.DataFrame(diff_matrix, index=models, columns=metrics)

    plt.figure(figsize=(10, 8))
    sns.heatmap(diff_df, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Difference (DPO - Semantic)'},
                linewidths=1, linecolor='gray')
    plt.title('Performance Difference: DPO vs Semantic\n(Positive = DPO Better, Negative = Semantic Better)',
              fontsize=14, fontweight='bold')
    plt.ylabel('Model', fontsize=12)
    plt.xlabel('Metric', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dpo_vs_semantic_difference_heatmap.png'), dpi=300)
    plt.close()

    print(f"✓ Difference heatmap saved")

def print_summary_stats(df):
    """Print summary statistics comparing semantic vs DPO"""

    df_filtered = df[df['Strategy'].isin(['semantic', 'dpo'])].copy()

    print("\n" + "="*60)
    print("SUMMARY STATISTICS: Semantic vs DPO")
    print("="*60)

    for strategy in ['semantic', 'dpo']:
        strategy_df = df_filtered[df_filtered['Strategy'] == strategy]
        print(f"\n{strategy.upper()} Strategy:")
        print(f"  Mean F1 Weighted: {strategy_df['F1 (Weighted)'].mean():.4f}")
        print(f"  Mean Accuracy:    {strategy_df['Accuracy'].mean():.4f}")
        print(f"  Mean F1 Macro:    {strategy_df['F1 (Macro)'].mean():.4f}")
        print(f"  Mean Latency:     {strategy_df['Avg Latency'].mean():.4f}s")

    print("\n" + "="*60)

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, 'experiment_results/classification/4_model-experiment-results', 'Full_model_comparison_results.yaml')
    output_dir = os.path.join(base_dir, 'experiment_results/classification/4_model-experiment-results', 'plots')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from: {data_path}")
    data = load_data(data_path)
    df = create_dataframe(data)

    # Set seaborn style
    sns.set_theme(style="whitegrid")

    print(f"\nGenerating plots...")
    print(f"Total entries: {len(df)}")
    print(f"Strategies: {df['Strategy'].unique()}")
    print(f"Models: {df['Model'].unique()}")

    plot_f1_weighted_comparison(df, output_dir)
    plot_f1_comparison_all_metrics(df, output_dir)
    plot_difference_heatmap(df, output_dir)

    print_summary_stats(df)

    print(f"\n✓ All plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
