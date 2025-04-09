from fpgrowth import main
from clean import clean_rules
from normalize_support import normalize_and_testing
import pandas as pd
import os
import sys

def count_rules(file_path):
    """Count the number of rules in a CSV file."""
    if os.path.exists(file_path):
        return len(pd.read_csv(file_path))
    return 0


def binary_search_min_support(train_df, path, class_label, top_n_higher):
    """Finds the optimal min_support using binary search."""
    low, high = 0.0, 1.0
    best_support = high  # Start with the highest support

    while high - low > 0.01:  # Stop when precision is small enough
        mid = (low + high) / 2

        # Run FP-Growth with current support for the given class
        print(f"Running FP-Growth with min_support: {mid}")
        main(train_df, mid, class_label, path)

        # Clean rules
        file_prefix = "pos" if class_label == 1 else "neg"
        clean_rules(f"{path}{file_prefix}_rules.csv",
                    f"{path}{file_prefix}_clean_rules.csv")

        # Count cleaned rules
        rule_count = count_rules(f"{path}{file_prefix}_clean_rules.csv")
        print(f"Found {rule_count} rules.",top_n_higher)

        if rule_count >= top_n_higher:
            print(f"Found FP-Growth with min_support: {mid}")
            best_support = mid  # Update best found support
            low = mid  # Try a higher min_support (reduce rules)
            break
        else:
            high = mid  # Try a lower min_support (increase rules)

    return best_support


def lad_fp(train_df, test_df, top_n_lower, top_n_higher, path):
    """Runs the LAD-FP pipeline using binary search to find the optimal min_support."""

    # Find best min_support separately for positive (1) and negative (0) rules
    print("POS BINARY SEARCH.....")
    pos_min_support = binary_search_min_support(
        train_df, path, 1, top_n_higher)
    print("NEG BINARY SEARCH.....")
    neg_min_support = binary_search_min_support(
        train_df, path, 0, top_n_higher)

    print(
        f"Optimal pos_min_support: {pos_min_support}, neg_min_support: {neg_min_support}")

    # Grid Search, Normalization, and Testing
    # normalize_and_testing(f"{path}pos_clean_rules.csv", f"{path}neg_clean_rules.csv",top_n_lower, top_n_higher, test_df, path)

    print("Pipeline completed successfully!")


# Define input parameters
data_file = f"../Dataset/{sys.argv[1]}/Preserved_Bins/{sys.argv[1]}_train.csv"
test_file = f"../Dataset/{sys.argv[1]}/Train_Test/{sys.argv[1]}_test.csv"

ds = f"{sys.argv[1]}"  # Give dataset name to make a folder


top_n_lower = 10
top_n_higher = 100

dataset = ds if ds else os.path.basename(data_file).split('.')[0]
os.makedirs(f"../Rules/{sys.argv[1]}/Original", exist_ok=True)
path = f"../Rules/{sys.argv[1]}/Original/"

train_df = pd.read_csv(data_file)
test_df = pd.read_csv(test_file)



# Run the pipeline
lad_fp(train_df, test_df,top_n_lower, top_n_higher, path)
