import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
for metric in ["ch_value", "h_value", "coverage","normalized_support", "oh_value"]:
    for top in range(10,110,10):
        for types in ["pos", "neg"]:

            # Load the dataset
            data = pd.read_csv(f'../Outputs/{sys.argv[1]}/{metric}/{top}/{types}_rules.csv')

            # Create a folder named "boxplots" if it doesn't exist
            output_folder = f'../Outputs/{sys.argv[1]}/{metric}/{top}/'
            os.makedirs(output_folder, exist_ok=True)

            # List of columns to plot
            columns_to_plot = ['h_value', 'ch_value', 'coverage', 'normalized_support', 'oh_value']

            # Generate and save a single boxplot for all columns
            plt.figure(figsize=(10, 6))
            data_to_plot = [data[column] for column in columns_to_plot]
            plt.boxplot(data_to_plot, vert=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
            plt.title(f'Boxplots of {metric} for {types} rules (Top {top})')
            plt.ylabel('Values')
            plt.xticks(ticks=range(1, len(columns_to_plot) + 1), labels=columns_to_plot)
            
            # Save the plot in the "boxplots" folder
            output_path = os.path.join(output_folder, f'{types}_rules columns_boxplot.png')
            plt.savefig(output_path)
            plt.close()  # Close the plot to avoid overlapping plots

        print(f"Boxplots have been saved in the '{output_folder}' folder.")