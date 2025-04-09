import pandas as pd
import matplotlib.pyplot as plt
import sys

# Read the CSV file
df = pd.read_csv(f'../Outputs/{sys.argv[1]}/all_metrics_results.csv')

# Plot
plt.figure(figsize=(12, 7))
for metric in df['metric'].unique():
    subset = df[df['metric'] == metric]
    plt.plot(subset['num_rules'], subset['accuracy'], marker='o', label=metric)

# Labels and legend
plt.title('Accuracy vs. Number of Rules by Metric')
plt.xlabel('Number of Rules')
plt.ylabel('Accuracy')
plt.legend(title='Metric')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"../Outputs/{sys.argv[1]}/Plots/plot.png", dpi=300, bbox_inches='tight')