from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import csv

# orig_file = ""

def evaluate_term(term, data_point):
    # Extract the feature and the condition
    if "cp" in term:
        term = term.strip()
        is_negated = term.startswith("~")
        if is_negated:
            term = term[1:]  # Remove the negation symbol
        feature, condition = term.split('cp')
        # print(feature,condition)
        feature = feature.strip()
        condition = condition.strip()
        if '>=' in condition:
            value = float(condition.split('>=')[1])
            return data_point[feature] >= value if not is_negated else data_point[feature]<value
        elif '<' in condition:
            value = float(condition.split('<')[1])
            return data_point[feature] < value if not is_negated else data_point[feature] >= value
        else:
            raise ValueError(f"Unsupported condition: {condition}")
    else:
        feature, value = term.split('_')
        feature = feature.strip()
        value = value.strip()
        return data_point[feature] == value

def testing(test_df, orig_df, pos_patterns_df, neg_patterns_df):
    result_df = pd.DataFrame()
    result_df["actual_result"] = orig_df["result"].copy()
    result_df["pred_result"] = None

    for index, row in test_df.iterrows():
        pos_score = 0
        neg_score = 0
        pos_count = 0
        neg_count = 0 
        # Evaluate positive patterns
        for _, pattern_row in pos_patterns_df.iterrows():
            pattern = pattern_row['rule'].split(' & ')
            if all(evaluate_term(term, row) for term in pattern):
                pos_score += pattern_row['normalized_support']
                pos_count += 1

        # Evaluate negative patterns
        for _, pattern_row in neg_patterns_df.iterrows():
            pattern = pattern_row['rule'].split(' & ')
            if all(evaluate_term(term, row) for term in pattern):
                neg_score += pattern_row['normalized_support']
                neg_count += 1
        
        # Decide prediction based on weighted scores
        if pos_score != 0 and neg_score == 0:
            result_df.at[index, "pred_result"] = 1
        elif pos_score == 0 and neg_score != 0:
            result_df.at[index, "pred_result"] = 0
        else:
            if pos_score > neg_score:
                result_df.at[index, "pred_result"] = 1
            elif pos_score < neg_score:
                result_df.at[index, "pred_result"] = 0
            else:
                result_df.at[index, "pred_result"] = -1
        # with open('difference_normalize.csv', 'a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow([pos_score-neg_score])   

    result_df['pred_result'] = result_df['pred_result'].astype(int)

    return result_df


def cm(pos_patterns_df, neg_patterns_df, orig_df, output_path):
    # orig_df = pd.read_csv(orig_file)  # 'Dataset/UNSW/UNSW_NB15_testing-set.csv'
    test_df = orig_df.drop('result', axis=1)
    print(test_df)
    result_df = testing(test_df, orig_df, pos_patterns_df, neg_patterns_df)
    # result_df.to_csv(
    #     f"{path}test_result_testing_normalize.csv", index=None)

    num_unknown_predictions = (result_df['pred_result'] == -1).sum()
    print(f"Number of unknown predictions: {num_unknown_predictions}")

    result_df['adjusted_pred_result'] = result_df.apply(
        lambda row: 0 if row['actual_result'] == 1 and row['pred_result'] == -1 else
        1 if row['actual_result'] == 0 and row['pred_result'] == -1 else
        row['pred_result'],
        axis=1
    )

    y_true = result_df['actual_result']
    y_pred = result_df['adjusted_pred_result']

    # Compute and display classification metrics
    precision = precision_score(y_true, y_pred, zero_division=1)  # Set to 1 or 0 based on your preference
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    accuracy = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cm = confusion_matrix(y_true, y_pred)

    metrics = (
        f"\n\nClassification Metrics\n"
        f"Precision: {precision}\n"
        f"Recall: {recall}\n"
        f"F1 Score: {f1}\n"
        f"Accuracy: {accuracy}\n"
        f"\nConfusion Matrix (Actual vs Predicted)\n"
        f"{'':<12} {'Predicted 0':<15} {'Predicted 1'}\n"
        f"{'Actual 0':<12} {cm[0, 0]:<15} {cm[0, 1]}\n"
        f"{'Actual 1':<12} {cm[1, 0]:<15} {cm[1, 1]}\n"
    )

    print(metrics)
    # Write metrics to a file
    # file_name = f"output_files/top_n_classification_metrics.txt"
    with open(output_path, "w") as file:
        file.write(metrics)

