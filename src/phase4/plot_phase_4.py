import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    return data

def create_dataframe(data):
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(data)
    return df

def plot_accuracy(df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Model', y='Accuracy', hue='Strategy')
    plt.title('Accuracy by Model and Strategy')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.ylim(0, 1.0)
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()

def plot_f1_scores(df, output_dir):
    # Melt dataframe to plot both F1 scores side by side if needed,
    # but separate plots might be cleaner or just one interesting one.
    # Let's do Weighted F1 primarily as it's often more representative for imbalanced if any.

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Model', y='F1 (Weighted)', hue='Strategy')
    plt.title('Weighted F1 Score by Model and Strategy')
    plt.ylabel('F1 Score (Weighted)')
    plt.xlabel('Model')
    plt.ylim(0, 1.0)
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_weighted_comparison.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Model', y='F1 (Macro)', hue='Strategy')
    plt.title('Macro F1 Score by Model and Strategy')
    plt.ylabel('F1 Score (Macro)')
    plt.xlabel('Model')
    plt.ylim(0, 1.0)
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_macro_comparison.png'))
    plt.close()

def plot_latency(df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Model', y='Avg Latency', hue='Strategy')
    plt.title('Average Latency by Model and Strategy')
    plt.ylabel('Latency (s)')
    plt.xlabel('Model')
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'latency_comparison.png'))
    plt.close()

def plot_tokens(df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Model', y='Avg Completion Tokens', hue='Strategy')
    plt.title('Average Completion Tokens by Model and Strategy')
    plt.ylabel('Avg Completion Tokens')
    plt.xlabel('Model')
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'completion_tokens_comparison.png'))
    plt.close()

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, 'experiment_results/classification/4_model-experiment-results', 'Full_model_comparison_results.yaml')
    output_dir = os.path.join(base_dir, 'experiment_results/classification/4_model-experiment-results', 'plots')

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    data = load_data(data_path)
    df = create_dataframe(data)

    # Set seaborn style
    sns.set_theme(style="whitegrid")

    print("Generating plots...")
    plot_accuracy(df, output_dir)
    plot_f1_scores(df, output_dir)
    plot_latency(df, output_dir)
    plot_tokens(df, output_dir)
    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    main()
