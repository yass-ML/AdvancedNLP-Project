
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.ticker import ScalarFormatter

def plot_scaling_scatter():
    # Load data
    yaml_path = os.path.join(os.path.dirname(__file__), "../../experiment_results/classification/5_K_scaling_experiment_results/Full_phase5_scaling_results.yaml")

    if not os.path.exists(yaml_path):
        print(f"Error: {yaml_path} not found")
        return

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    df = pd.DataFrame(data)
    df = df[df['Status'] == 'Success']

    # Sort to ensure lines connect K in order
    df = df.sort_values(by=['Model', 'K'])

    # Setup styling
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 10))

    # Create unique colors for models
    models = df['Model'].unique()
    palette = sns.color_palette("tab10", n_colors=len(models))
    model_colors = dict(zip(models, palette))

    # 1. Plot connected lines (Trajectories)
    sns.lineplot(data=df, x='Avg_Latency', y='F1_Weighted', hue='Model',
                 palette=model_colors, linewidth=1.5, alpha=0.6, sort=False, legend=False)

    # 2. Plot scatter points
    sns.scatterplot(data=df, x='Avg_Latency', y='F1_Weighted', hue='Model',
                    palette=model_colors, s=100, edgecolor='black', alpha=0.9, legend='full')

    # 3. Annotate K values
    # To avoid clutter, we can try to be smart, or just annotate all since requested.
    # We'll annotate all but use a small font.
    for i, row in df.iterrows():
        plt.text(row['Avg_Latency'], row['F1_Weighted'], f"K={row['K']}",
                 fontsize=8, ha='right', va='bottom', fontweight='bold', alpha=0.8)

    # 4. Axes and Labels
    plt.xscale('log')
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())

    plt.title('Phase 5 Scaling: F1 Weighted vs Latency vs K', fontsize=18)
    plt.xlabel('Average Latency (seconds) [Log Scale]', fontsize=14)
    plt.ylabel('F1 Weighted', fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    # Improve Legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Model', borderaxespad=0.)

    plt.tight_layout()

    output_dir = os.path.join(os.path.dirname(__file__), "../../graphs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "phase5_scaling_scatter.png")

    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    plot_scaling_scatter()
