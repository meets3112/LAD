import pandas as pd
import gc
from joblib import Parallel, delayed
import sys

# FUNCTION TO BINARIZE STRING COLUMNS MANUALLY


def binarize_string_columns(df, string_columns):
    """Manually binarizes string columns by creating new binary columns."""
    binarized = pd.DataFrame()
    nominal_count = 0  # Counter for binarized string columns

    for col_name in string_columns:
        col_data = df[col_name]
        unique_values = col_data.dropna().unique()

        for value in unique_values:
            binarized[f"{col_name}_{value}"] = (col_data == value).astype(int)
            nominal_count += 1  # Increment for each new binary column

    return binarized, nominal_count

# FUNCTION TO PROCESS EACH NUMERICAL COLUMN


def process_column(col, df, result_col):
    cutpoints = []
    collst = [col, result_col]

    df_short = pd.DataFrame(df, columns=collst)
    df_short.sort_values(col, ascending=False, inplace=True)
    temp = df_short[[col, result_col]].values.T.tolist()

    # CHANGE LABEL IF SAME OBSERVATION HAS DIFFERENT LABEL
    for i in range(len(temp[0]) - 1):
        if (temp[0][i] == temp[0][i + 1]) and (temp[1][i] != temp[1][i + 1]):
            c = max(temp[1]) + 1
            temp[1][i] = c
            temp[1][i + 1] = c

    # CUTPOINT CALCULATION
    for i in range(len(temp[0]) - 1):
        if (temp[0][i] != temp[0][i + 1]) and (temp[1][i] != temp[1][i + 1]):
            cutpoints.append((temp[0][i] + temp[0][i + 1]) / 2)

    return col, cutpoints

# MAIN FUNCTION


def binarization(df, result_col,path):
    original_df = df.copy()

    # Identify numerical and string columns
    num_columns = df.select_dtypes(
        include=['int64', 'float64']).columns.tolist()
    string_columns = df.select_dtypes(include=['object']).columns.tolist()

    print(num_columns)
    print(string_columns)
    # Exclude result column from processing
    if result_col in num_columns:
        num_columns.remove(result_col)
    if result_col in string_columns:
        string_columns.remove(result_col)

    # Process numerical columns
    all_cutpoints = []
    results = Parallel(n_jobs=-1)(delayed(process_column)
                                  (col, df, result_col) for col in num_columns)

    # Collect cutpoints
    all_cutpoints = [cutpoints for _, cutpoints in results]
    count_cutpoint = sum(len(cutpoints) for cutpoints in all_cutpoints)
    print(f"Total numerical cutpoints: {count_cutpoint}")

    # BINARIZATION FOR NUMERICAL COLUMNS
    result_list = original_df[result_col].tolist()
    df1 = original_df[num_columns]  # Keep only numerical columns

    temp = df1.values.T.tolist()
    col_names = df1.columns

    del df1
    gc.collect()

    all_bin = []
    lst = []
    ctr = 0  # Counter for column names
    for l in all_cutpoints:
        # if len(l) < 175:
        for i in l:
            # Include column name
            lst.append(col_names[ctr] + "cp=" + str(i))
            bin_d = [(1 if j > i else 0) for j in temp[ctr]]
            all_bin.append(bin_d)
        # else:
        #     print(col_names[ctr], "  ", len(l))
        ctr += 1
    print("Number of numerical cutpoints included:", len(lst))

    level_df = pd.DataFrame(all_bin).transpose()
    level_df.columns = lst

    # BINARIZATION FOR STRING COLUMNS (Manually)
    binarized_strings, nominal_count = binarize_string_columns(
        original_df, string_columns)
    print(f"Total nominal (string) binarized columns: {nominal_count}")

    # MERGE STRING AND NUMERICAL BINARIZATION
    final_bin_df = pd.concat([level_df, binarized_strings], axis=1)
    final_bin_df['label'] = result_list

    final_bin_df.to_csv(path, index=None)
    return final_bin_df


# Load data and run
df = pd.read_csv(f'../Dataset/{sys.argv[1]}/Train_Test/{sys.argv[1]}_train.csv')
bin_data = binarization(df, "label", f"../Dataset/{sys.argv[1]}/Binarized/{sys.argv[1]}_train.csv")
print("Binarization completed.")

















# import pandas as pd
# import gc
# from joblib import Parallel, delayed

# # FUNCTION TO BINARIZE STRING COLUMNS MANUALLY


# def binarize_string_columns(df, string_columns):
#     """Manually binarizes string columns by creating new binary columns."""
#     binarized = pd.DataFrame()
#     nominal_count = 0  # Counter for binarized string columns

#     for col_name in string_columns:
#         col_data = df[col_name]
#         unique_values = col_data.dropna().unique()

#         for value in unique_values:
#             binarized[f"{col_name}_{value}"] = (col_data == value).astype(int)
#             nominal_count += 1  # Increment for each new binary column

#     return binarized, nominal_count

# # FUNCTION TO PROCESS EACH NUMERICAL COLUMN


# def process_column(col, df, result_col):
#     cutpoints = []
#     collst = [col, result_col]

#     # Use copy() to prevent SettingWithCopyWarning
#     df_short = df[collst].copy()
#     df_short = df_short.sort_values(col, ascending=False)

#     temp = df_short[[col, result_col]].to_numpy().T.tolist()

#     # CHANGE LABEL IF SAME OBSERVATION HAS DIFFERENT LABEL
#     for i in range(len(temp[0]) - 1):
#         if (temp[0][i] == temp[0][i + 1]) and (temp[1][i] != temp[1][i + 1]):
#             c = max(temp[1]) + 1
#             temp[1][i] = c
#             temp[1][i + 1] = c

#     # CUTPOINT CALCULATION
#     for i in range(len(temp[0]) - 1):
#         if (temp[0][i] != temp[0][i + 1]) and (temp[1][i] != temp[1][i + 1]):
#             cutpoints.append((temp[0][i] + temp[0][i + 1]) / 2)

#     print(col, "  ", len(cutpoints))
#     return col, cutpoints

# # MAIN FUNCTION


# def binarization(df, result_col):
#     original_df = df.copy()

#     # Identify numerical and string columns explicitly
#     num_columns = df.select_dtypes(include=['number']).columns.tolist()
#     string_columns = df.select_dtypes(include=['object']).columns.tolist()

#     # Print unique value counts for each column
#     for col in df.columns:
#         print(f"{col}: {df[col].nunique()} unique values")

#     print("String Columns:", string_columns)

#     # Exclude result column from processing
#     if result_col in num_columns:
#         num_columns.remove(result_col)
#     if result_col in string_columns:
#         string_columns.remove(result_col)

#     print("Numerical Columns:", num_columns)

#     # Process numerical columns using joblib (parallel processing)
#     results = Parallel(n_jobs=-1, backend='loky')(
#         delayed(process_column)(col, df, result_col) for col in num_columns
#     )

#     # Collect cutpoints
#     all_cutpoints = [cutpoints for _, cutpoints in results]
#     count_cutpoint = sum(len(cutpoints) for cutpoints in all_cutpoints)
#     print(f"Total numerical cutpoints: {count_cutpoint}")

#     # BINARIZATION FOR NUMERICAL COLUMNS
#     result_list = original_df[result_col].tolist()
#     # Ensure only numerical columns are kept
#     df1 = original_df[num_columns].copy()

#     temp = df1.to_numpy().T.tolist()
#     col_names = df1.columns

#     del df1
#     gc.collect()

#     all_bin = []
#     lst = []
#     ctr = 0  # Counter for column names
#     for l in all_cutpoints:
#         for i in l:
#             lst.append(f"{col_names[ctr]}cp={i}")  # Include column name
#             bin_d = [(1 if j > i else 0) for j in temp[ctr]]
#             all_bin.append(bin_d)
#         ctr += 1

#     print("Number of numerical cutpoints included:", len(lst))

#     level_df = pd.DataFrame(all_bin).transpose()
#     level_df.columns = lst

#     # BINARIZATION FOR STRING COLUMNS (Manually)
#     binarized_strings, nominal_count = binarize_string_columns(
#         original_df, string_columns)
#     print(f"Total nominal (string) binarized columns: {nominal_count}")

#     # MERGE STRING AND NUMERICAL BINARIZATION
#     final_bin_df = pd.concat([level_df, binarized_strings], axis=1)
#     final_bin_df['result'] = result_list

#     return final_bin_df


# # Load data and run
# # Read everything as str to avoid dtype issues
# df = pd.read_csv("Extra Dataset/Dataset/Heart Disease/heart.csv")
# # Convert numeric columns after reading
# df = df.apply(pd.to_numeric, errors='ignore')

# bin_data = binarization(df, "result")

# # Save final binarized data to CSV
# bin_data.to_csv("Extra Dataset/Dataset/Heart Disease/heart_ss.csv", index=False)
