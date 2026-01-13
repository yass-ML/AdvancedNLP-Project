import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data extracted from your results
data = [
    # Llama-3-8B
    {"Model": "Llama-3-8B", "K": 1, "Accuracy": 0.68},
    {"Model": "Llama-3-8B", "K": 3, "Accuracy": 0.76},
    {"Model": "Llama-3-8B", "K": 5, "Accuracy": 0.71},
    {"Model": "Llama-3-8B", "K": 10, "Accuracy": 0.74},
    {"Model": "Llama-3-8B", "K": 15, "Accuracy": 0.76},
    {"Model": "Llama-3-8B", "K": 20, "Accuracy": 0.77},
    {"Model": "Llama-3-8B", "K": 25, "Accuracy": 0.75},

    # Qwen3-8B (Reasoning)
    {"Model": "Qwen3-8B", "K": 1, "Accuracy": 0.72},
    {"Model": "Qwen3-8B", "K": 3, "Accuracy": 0.82}, # Peak
    {"Model": "Qwen3-8B", "K": 5, "Accuracy": 0.74},
    {"Model": "Qwen3-8B", "K": 10, "Accuracy": 0.73},
    {"Model": "Qwen3-8B", "K": 20, "Accuracy": 0.67}, # Drop
    {"Model": "Qwen3-8B", "K": 25, "Accuracy": 0.64},

    # Mistral-7B
    {"Model": "Mistral-7B", "K": 1, "Accuracy": 0.60},
    {"Model": "Mistral-7B", "K": 3, "Accuracy": 0.68},
    {"Model": "Mistral-7B", "K": 5, "Accuracy": 0.75}, # Peak
    {"Model": "Mistral-7B", "K": 10, "Accuracy": 0.73},
    {"Model": "Mistral-7B", "K": 20, "Accuracy": 0.70},
    {"Model": "Mistral-7B", "K": 25, "Accuracy": 0.73},

    # Phi-3-Mini (Collapse)
    {"Model": "Phi-3-Mini", "K": 1, "Accuracy": 0.22},
    {"Model": "Phi-3-Mini", "K": 3, "Accuracy": 0.39},
    {"Model": "Phi-3-Mini", "K": 5, "Accuracy": 0.32},
    {"Model": "Phi-3-Mini", "K": 10, "Accuracy": 0.01}, # Collapse
]

df = pd.DataFrame(data)

# Plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Draw Lines
sns.lineplot(
    data=df, 
    x="K", 
    y="Accuracy", 
    hue="Model", 
    style="Model", 
    markers=True, 
    dashes=False, 
    linewidth=2.5,
    markersize=9
)

# Highlights
plt.title("Impact of Shot Count (K) on Model Accuracy", fontsize=16, fontweight='bold', pad=15)
plt.ylabel("Classification Accuracy", fontsize=12, fontweight='bold')
plt.xlabel("Number of Shots (K)", fontsize=12, fontweight='bold')
plt.ylim(0, 0.9)
plt.xlim(0, 26)

# Annotate Qwen Peak
plt.annotate('Reasoning Peak (k=3)', xy=(3, 0.82), xytext=(6, 0.85),
             arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10, fontweight='bold')

# Annotate Phi-3 Collapse
plt.annotate('Context Collapse', xy=(10, 0.01), xytext=(12, 0.15),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=10, fontweight='bold', color='red')

plt.legend(title="Model Architecture", loc='lower right')
plt.tight_layout()

# Save
plt.savefig("k_shot_analysis.png", dpi=300)