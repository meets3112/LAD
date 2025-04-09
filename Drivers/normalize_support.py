import pandas as pd
from combined_sum_testing import cm
import os

# orig_file = ""
# orig_df = pd.read_csv(pos_file)


def normalize_helper(df, path):
    # df = df.head(n).reset_index(drop=True)
    # Calculate the sum of 'h_value'
    total_support_value = df['normalized_support'].sum()

    # Normalize the 'h_value' column
    df['normalized_support'] = df['normalized_support'] / total_support_value

    # Save the normalized data back to a new CSV file
    df.to_csv(path, index=False)
    return df


def normalize_and_testing(pos_file, neg_file, top_n_lower, top_n_higher, orig_df, path):
    pos_rules = pd.read_csv(pos_file)
    neg_rules = pd.read_csv(neg_file)

    for i in range(top_n_lower, top_n_higher + 1, 10):
        print(f"Testing for top {i} rules.")
        folder_path = f"{path}{i}/"
        os.makedirs(folder_path, exist_ok=True)  # Ensure subfolder exists

        pos_patterns_df = pos_rules.head(i).reset_index(drop=True)
        neg_patterns_df = neg_rules.head(i).reset_index(drop=True)

        pos_save = normalize_helper(
            pos_patterns_df, f"{folder_path}pos_rules.csv")
        neg_save = normalize_helper(
            neg_patterns_df, f"{folder_path}neg_rules.csv")

        cm(pos_save, neg_save, orig_df, f"{folder_path}cm.txt")

    # Normalize the data
    # pos_rules = normalize_helper(
    #     pos_rules, n, f'{path}pos_final_top{n}_normalize.csv')
    # neg_rules = normalize_helper(
    #     neg_rules, n, f'{path}neg_final_top{n}_normalize.csv')
    # return df1, df2


