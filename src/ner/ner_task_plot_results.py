import yaml
import matplotlib.pyplot as plt
import argparse
import numpy as np

def plot_scaling_results(yaml_file="ner_task_scaling_results.yaml", output_file="ner_scaling_plot.png"):
    # Read YAML
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    results = data.get('scaling_experiment', [])
    if not results:
        print("No results found in YAML.")
        return

    # Group by model
    models = {}
    for res in results:
        m = res['model']
        if m not in models:
            models[m] = {'latency': [], 'f1': [], 'k': []}

        models[m]['latency'].append(res['avg_latency'])
        models[m]['f1'].append(res['f1'])
        models[m]['k'].append(res['k'])

    # Plot
    plt.figure(figsize=(12, 8))

    # Colors/Markers similar to reference if possible, or standard matplotlib cycle
    # Reference had: Blue, Orange, Green, Red, Purple, Brown
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, (model_name, values) in enumerate(models.items()):
        # Sort by K to ensure line connects properly
        zipped = sorted(zip(values['k'], values['latency'], values['f1']))
        ks, lats, f1s = zip(*zipped)

        # Plot Line
        plt.plot(lats, f1s, linestyle='-', linewidth=1.5, alpha=0.7, color=colors[i], zorder=1)

        # Plot Scatter Points with black edges
        plt.scatter(lats, f1s, s=100, color=colors[i], edgecolors='black', linewidth=1, label=model_name, zorder=2)

        # Annotate points with K values
        for k, lat, f1 in zip(ks, lats, f1s):
            plt.text(lat, f1 + 0.015, f"K={k}", fontsize=8, ha='center', va='bottom', fontweight='bold', alpha=0.9, zorder=3)

    # Styling
    plt.xscale('log')
    plt.xlabel('Average Latency (seconds) [Log Scale]')
    plt.ylabel('F1 Score')
    plt.title('Phase 5 Scaling: F1 Score vs Latency vs K')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Ensure proper log ticks formatting if needed, but default log scale usually works well.
    # We can use ScalarFormatter for x-axis if we want non-scientific notation.
    from matplotlib.ticker import ScalarFormatter
    plt.gca().xaxis.set_major_formatter(ScalarFormatter())

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", default="ner_task_scaling_results.yaml")
    parser.add_argument("--output", default="ner_scaling_plot.png")
    args = parser.parse_args()

    plot_scaling_results(args.yaml, args.output)
