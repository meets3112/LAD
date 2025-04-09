import pandas as pd
import sys

# Load the datasets
heart_train = pd.read_csv(f"../Dataset/{sys.argv[1]}/Train_Test/{sys.argv[1]}_train.csv")
heart_train_ss = pd.read_csv(f"../Dataset/{sys.argv[1]}/Support_Set/{sys.argv[1]}_train.csv")

# Get the columns from both datasets
ss_columns = set(heart_train_ss.columns)
# Filter only binary columns from the original dataset
binary_columns = {col for col in heart_train.columns if heart_train[col].dropna().isin([0, 1]).all()}


# Extract the base column names from ss_columns (e.g., "col" from "col={some_num}")
ss_base_columns = {col.split('=')[0] for col in ss_columns}

# Find the removed columns
removed_columns = binary_columns - ss_base_columns

# Add the removed columns to heart_train_ss with renamed columns
for col in removed_columns:
        heart_train_ss[f"{col}cp=0.5"] = heart_train[col]

result_col = heart_train_ss.pop('label')
heart_train_ss['label'] = result_col
heart_train_ss.to_csv(f"../Dataset/{sys.argv[1]}/Preserved_Bins/{sys.argv[1]}_train.csv", index=False)
print("New CSV with renamed removed columns added has been created: heart_train_ss_with_removed_cols.csv")