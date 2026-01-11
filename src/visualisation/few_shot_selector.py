import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

data = [
    {"Strategy": "Random", "Accuracy": 50.0, "Latency": 0.31, "Type": "Baseline"}, 
    {"Strategy": "Lexical (BM25)", "Accuracy": 71.0, "Latency": 0.32, "Type": "Traditional"}, 
    {"Strategy": "Semantic (Bi-Encoder)", "Accuracy": 76.0, "Latency": 0.30, "Type": "SOTA"},
    {"Strategy": "Cross-Encoder", "Accuracy": 69.0, "Latency": 0.95, "Type": "Experimental"},
    {"Strategy": "DPO (Hybrid Selector)", "Accuracy": 76.0, "Latency": 0.96, "Type": "Experimental"},
]

df = pd.DataFrame(data)

fig, ax1 = plt.subplots(figsize=(12, 7))
sns.set_style("white")

colors = sns.color_palette("viridis", n_colors=len(df))
bars = sns.barplot(
    data=df, 
    x="Strategy", 
    y="Accuracy", 
    ax=ax1, 
    palette=colors, 
    alpha=0.8,
    edgecolor="black",
    linewidth=1.5
)

ax1.set_ylabel("Classification Accuracy (%)", fontsize=14, fontweight='bold', color='#333333')
ax1.set_xlabel("Selection Strategy", fontsize=14, fontweight='bold')
ax1.set_ylim(40, 85)
ax1.tick_params(axis='y', labelsize=12)
ax1.tick_params(axis='x', labelsize=11, rotation=0)

for i, bar in enumerate(bars.patches):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width()/2., 
        height + 0.5, 
        f'{height:.1f}%', 
        ha='center', 
        va='bottom', 
        fontsize=12, 
        fontweight='bold',
        color='black'
    )

ax2 = ax1.twinx()
sns.lineplot(
    data=df, 
    x="Strategy", 
    y="Latency", 
    ax=ax2, 
    color='#e74c3c',
    marker='o', 
    markersize=12, 
    linewidth=3, 
    label='Latency (s)'
)

ax2.set_ylabel("Avg Latency (seconds)", fontsize=14, fontweight='bold', color='#e74c3c')
ax2.set_ylim(0, 1.3)
ax2.tick_params(axis='y', labelcolor='#e74c3c', labelsize=12)
ax2.grid(False)

for i, txt in enumerate(df.Latency):
    ax2.text(
        i, 
        txt + 0.08,
        f'{txt}s', 
        ha='center', 
        va='bottom', 
        fontsize=11, 
        fontweight='bold', 
        color='#e74c3c',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
    )

plt.title("Comparison of Strategies: Accuracy (Bars) vs. Latency (Line)", fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()

plt.savefig("strategy_comparison_barchart.png", dpi=300)
plt.show()