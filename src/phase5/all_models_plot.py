import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D
import os
import yaml

# 1. Load Data from YAML
def load_data():
    yaml_path = os.path.join(os.path.dirname(__file__), "../../experiment_results/classification/5_K_scaling_experiment_results/Full_phase5_scaling_results.yaml")

    if not os.path.exists(yaml_path):
        print(f"Error: Results file not found at {yaml_path}")
        return []

    with open(yaml_path, 'r') as f:
        raw_data = yaml.safe_load(f)
    print(f"Loaded {len(raw_data)} entries from YAML.")
    return raw_data

raw_data = load_data()

data = []
model_name_map = {
    'llama3:8b': 'Llama 3',
    'mistral:7b': 'Mistral 7B',
    'gemma:7b': 'Gemma 7B',
    'phi3:mini': 'Phi-3 Mini',
    'qwen2:7b': 'Qwen2 7B',
    'qwen3:8b': 'Qwen3 8B'
}

strategy_map = {
    'dpo': 'DPO',
    'semantic': 'Semantic',
    'random': 'Random',
    'lexical': 'Lexical',
    'cross_encoder': 'Cross Encoder'
}

for entry in raw_data:
    if entry.get('Status') != 'Success':
        continue

    m_name = entry.get('Model', 'Unknown')
    clean_name = model_name_map.get(m_name, m_name)

    strat = entry.get('Strategy', 'Unknown')
    clean_strat = strategy_map.get(strat, strat.capitalize())

    data.append({
        'Model': clean_name,
        'Strategy': clean_strat,
        'K': entry.get('K'),
        'Accuracy': entry.get('Accuracy'),
        'Latency': entry.get('Avg_Latency')
    })

df = pd.DataFrame(data)
output_dir = os.path.join(os.path.dirname(__file__), "../../graphs")
os.makedirs(output_dir, exist_ok=True)

if df.empty:
    print("No success data found to plot.")
    exit()

plt.figure(figsize=(14, 9))
plt.rcParams['font.family'] = 'sans-serif'

colors = {
    'Llama 3': '#1f77b4',   # Bleu
    'Mistral 7B': '#ff7f0e',# Orange
    'Gemma 7B': '#2ca02c',  # Vert
    'Phi-3 Mini': '#d62728',# Rouge
    'Qwen2 7B': '#9467bd',  # Violet
    'Qwen3 8B': '#8c564b'   # Marron
}
markers = {'Semantic': 's', 'DPO': 'o'}

llama_df = df[df['Model'] == 'Llama 3'].sort_values('K')
if not llama_df.empty:
    plt.plot(llama_df['Latency'], llama_df['Accuracy'],
             color=colors.get('Llama 3', 'blue'), linestyle='--', linewidth=2, alpha=0.5, zorder=1, label='Llama 3 Scaling')

    for _, row in llama_df.iterrows():
        plt.text(row['Latency'], row['Accuracy'] + 0.01, f"K={row['K']}",
                 color=colors.get('Llama 3', 'blue'), fontsize=9, ha='center', fontweight='bold')

for _, row in df.iterrows():
    c = colors.get(row['Model'], 'gray')
    m = markers.get(row['Strategy'], 'o')

    plt.scatter(row['Latency'], row['Accuracy'],
                color=c, marker=m, s=150, edgecolors='white', linewidth=1.5, zorder=3, alpha=0.9)

    if row['Model'] != 'Llama 3':
        label_text = f"{row['Model']}\n({row['Strategy']})"

        xytext = (0, -25) # Par défaut: en dessous
        if 'Qwen' in row['Model']:
            xytext = (-30, 0) if row['Strategy'] == 'Semantic' else (30, 0)
        elif 'Phi3' in row['Model'] or 'Phi-3' in row['Model']:
            xytext = (0, -20)
        elif 'Mistral' in row['Model'] and row['Strategy'] == 'Semantic':
            xytext = (0, 15) # Au dessus

        plt.annotate(label_text, (row['Latency'], row['Accuracy']),
                     xytext=xytext, textcoords='offset points', ha='center', fontsize=8, alpha=0.8,
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.6))

plt.xscale('log')
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.grid(True, which="both", ls="-", alpha=0.15)

plt.xlabel('Average Latency (seconds) [Log Scale]', fontsize=12, fontweight='bold')
plt.ylabel('Accuracy (Test Set)', fontsize=12, fontweight='bold')
plt.title('Pareto Frontier: Accuracy vs. Cost (Latency)', fontsize=16)

legend_elements = [
    Line2D([0], [0], color=colors.get('Llama 3', 'blue'), lw=2, linestyle='--', label='Llama 3 Scaling'),
    Line2D([0], [0], marker='o', color='gray', label='DPO Strategy', markersize=10, lw=0),
    Line2D([0], [0], marker='s', color='gray', label='Semantic Strategy', markersize=10, lw=0),
]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
output_path = os.path.join(output_dir, "all_models_plot.png")
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
