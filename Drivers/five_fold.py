# import pandas as pd
# from sklearn.model_selection import KFold
# import os
# from main import lad_fp
# # from Binarization import num  # Import the binarization function
# from sneha_support import ss

# # Load dataset
# data_file = "Dataset/bank_additional_string/bank-additional-full.csv"
# data = pd.read_csv(data_file)

# # Add row_id column
# data.insert(0, 'row_id', range(1, len(data) + 1))

# # Define output path
# dataset_name = os.path.splitext(os.path.basename(data_file))[0] or "Unknown"
# output_folder = f"Dataset/{dataset_name}/"
# os.makedirs(output_folder, exist_ok=True)


# # Binarize the dataset
# result_column = "result"  # Replace with your actual result column name
# # Preserve row_id and binarize the remaining columns
# print("Binarization in progress...")
# # binarized_data = num(data.drop(columns=["row_id"]), result_column)
# bin_file = "Dataset/bank_additional_string/bank-additional-full_bin.csv"
# binarized_data = pd.read_csv(bin_file)

# print("Binarization completed.")
# # Add row_id back at index 0
# binarized_data.insert(0, "row_id", data["row_id"].values)
# # binarized_data.to_csv(f"{output_folder}binarized_data.csv", index=False)


# # Perform 5-fold split
# kf = KFold(n_splits=5, shuffle=True, random_state=42)

# for fold, (train_index, test_index) in enumerate(kf.split(binarized_data), start=1):
#     folder_path = f"{output_folder}fold{fold}/"
#     os.makedirs(folder_path, exist_ok=True)

#     # Get the row_ids of the test data and update it from the original data
#     # test_row_ids = binarized_data.iloc[test_index]['row_id'].values

#     # Corrected: Exclude rows from train_data_bin instead of including test_row_ids
#     train_data_bin = binarized_data.iloc[train_index].set_index("row_id")
#     test_data = data[~data['row_id'].isin(train_data_bin.index)]
    
#     test_data.to_csv(f"{folder_path}test_Data_{fold}.csv", index=False)

#     # Support Set Generation
#     train_data_bin = train_data_bin.reset_index(drop=True)
#     train_data_bin = train_data_bin.drop(columns=["row_id"], errors="ignore")
#     print("Support Set Generation in progress...")
#     test_data = test_data.drop(columns=["row_id"], errors="ignore")
#     train_data_ss = ss(train_data_bin, f"{folder_path}support_set.csv")
#     print("Support Set Generation Completed.")

#     top_n_lower = 10
#     top_n_higher = 100
    
#     lad_fp(train_data_ss, test_data, top_n_lower, top_n_higher, folder_path)

# print("Five-fold dataset split completed!")

import pandas as pd
from sklearn.model_selection import KFold
import os
from Drivers.main import lad_fp
from Drivers.sneha_support import ss
from Drivers.binarization import binarization

# Load dataset
data_file = "Dataset/Iris/iris.csv"
data = pd.read_csv(data_file)
result_col = "result"

# Define output path
dataset_name = os.path.splitext(os.path.basename(data_file))[0] or "Unknown"
output_folder = f"Dataset/{dataset_name}/"
os.makedirs(output_folder, exist_ok=True)

# Load pre-binarized data
bin_file = output_folder+dataset_name+"_bin.csv"
binarized_data = binarization(data, result_col, bin_file)

# Add row_id column if not already present
if "row_id" not in data.columns:
    data.insert(0, 'row_id', range(1, len(data) + 1))

# Ensure row_id is added back only if not already present
if "row_id" not in binarized_data.columns:
    binarized_data.insert(0, "row_id", data["row_id"].values)

# Perform 5-fold split
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_index, test_index) in enumerate(kf.split(binarized_data), start=1):
    folder_path = f"{output_folder}fold{fold}/"
    os.makedirs(folder_path, exist_ok=True)

    # Train and test split using correct `row_id`
    train_data_bin = binarized_data.iloc[train_index].copy()
    test_data = data.merge(binarized_data.loc[test_index, ["row_id"]], on="row_id", how="inner")

    # Debugging prints (optional)
    print(f"\nFold {fold}")
    print(f"Train size: {len(train_data_bin)}, Test size: {len(test_data)}")

    # Save test data only if it doesn't already exist
    test_file_path = f"{folder_path}test_Data_{fold}.csv"
    if not os.path.exists(test_file_path):
        test_data.to_csv(test_file_path, index=False)

    # Save train data only if it doesn't already exist
    train_bin_path = f"{folder_path}train_bin_{fold}.csv"
    if not os.path.exists(train_bin_path):
        train_data_bin.to_csv(train_bin_path, index=False)

    # Support Set Generation
    train_data_bin = train_data_bin.drop(columns=["row_id"], errors="ignore")
    print(train_data_bin.columns)
    test_data = test_data.drop(columns=["row_id"], errors="ignore")

    print("Support Set Generation in progress...")
    support_set_path = f"{folder_path}support_set.csv"
    if not os.path.exists(support_set_path):
        train_data_ss = ss(train_data_bin, support_set_path)
    else:
        train_data_ss = pd.read_csv(support_set_path)  # Load existing support set if it exists

    print("Support Set Generation Completed.")

    top_n_lower = 10
    top_n_higher = 100

    # Run LAD-FP Algorithm
    lad_fp(train_data_ss, test_data, top_n_lower, top_n_higher, folder_path)

print("Five-fold dataset split completed!")
