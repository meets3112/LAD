import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
# Load the dataset
neg_rules_data = pd.read_csv(f'../Rules/{sys.argv[1]}/All_Weights/neg_rules.csv')
# min = neg_rules_data['h_value'].min()
# max = neg_rules_data['h_value'].max()
# neg_rules_data['normalized_h_value'] = neg_rules_data['h_value'] - min / (max - min)
# neg_rules_data.to_csv('neg_new2.csv', index=False)


# Create a scatter plot
plt.figure(figsize=(8, 6))
# sns.scatterplot(x='h_value', y='coverage', data=neg_rules_data, alpha=0.7, hue='ch_value')
scatter = plt.scatter(neg_rules_data['h_value'], neg_rules_data['coverage'], c=neg_rules_data['ch_value'], cmap='viridis', alpha=0.8)
plt.colorbar(scatter, label='ch-value')
# plt.scatter(neg_rules_data['h_value'], neg_rules_data['coverage'], alpha=0.7, color='blue')

# Add labels and title
plt.xlabel('h-value')
plt.ylabel('Coverage')
plt.title('Comparison of Normalized Support and Coverage')

# Add a diagonal line for reference (if they are the same, points will lie on this line)
# plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='y = x (Equal Values)')

# Add legend
# plt.legend()

# Show the plot
plt.show()