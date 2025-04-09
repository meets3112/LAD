import pandas as pd
import os
import sys

# Define the base directory
base_dir = f"../Outputs/{sys.argv[1]}/"

# Define the folders and their corresponding metric names
folder_metrics = {
    "normalized_support": "normalized_support",
    "coverage": "coverage",
    "h_value": "h_support",
    "ch_value": "ch_support",
    "oh_value": "oh_support"
}

# Initialize an empty list to store all results
all_results = []

# Read each accuracy summary file
for folder, metric in folder_metrics.items():
    file_path = os.path.join(base_dir, folder, "accuracy_summary.csv")
    
    if os.path.exists(file_path):
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Add a column for the metric
        df['metric'] = metric
        
        # Reorder columns to have metric first
        df = df[['metric', 'num_rules', 'accuracy']]
        
        # Append to our results list
        all_results.append(df)
    else:
        print(f"File not found: {file_path}")

# Combine all dataframes
if all_results:
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Sort by metric and num_rules
    combined_df = combined_df.sort_values(['metric', 'num_rules'])
    
    # Save to a combined CSV file
    output_path = os.path.join(base_dir, "all_metrics_results.csv")
    combined_df.to_csv(output_path, index=False)
    
    print(f"Combined results saved to: {output_path}")
else:
    print("No results were found to combine")