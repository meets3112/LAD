import pandas as pd
import sys
def double_columns_except_result(df):
    new_df = df.copy()
    result_data = new_df['label']
    new_df.drop(columns=['label'], inplace=True)
    for col in new_df.columns:
            new_col = f"!{col}"
            new_df[new_col] = 1 - new_df[col]
    new_df['label'] = result_data
    return new_df

df = pd.read_csv(f"../Dataset/{sys.argv[1]}/Preserved_Bins/{sys.argv[1]}_train.csv")
df = double_columns_except_result(df)
df.to_csv(f"../Dataset/{sys.argv[1]}/Complemented/{sys.argv[1]}_train.csv", index=False)
print(f"New CSV with doubled columns has been created: {sys.argv[1]}_train.csv")