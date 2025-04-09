from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
import os
import matplotlib.pyplot as plt
import sys
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

def testing(test_df, orig_df, pos_patterns_df, neg_patterns_df, metric):
    result_df = pd.DataFrame()
    result_df["actual_result"] = orig_df["label"].copy()
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
                pos_score += pattern_row[f'normalized_{metric}']
                pos_count += 1

        # Evaluate negative patterns
        for _, pattern_row in neg_patterns_df.iterrows():
            pattern = pattern_row['rule'].split(' & ')
            if all(evaluate_term(term, row) for term in pattern):
                neg_score += pattern_row[f'normalized_{metric}']
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


def cm(pos_patterns_df, neg_patterns_df, orig_df, output_path, metric):
    # orig_df = pd.read_csv(orig_file)  # 'Dataset/UNSW/UNSW_NB15_testing-set.csv'
    test_df = orig_df.drop('label', axis=1)
    print(test_df)
    result_df = testing(test_df, orig_df, pos_patterns_df, neg_patterns_df, metric)
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

def normalize_helper(df, path, metric):
    # df = df.head(n).reset_index(drop=True)
    # Calculate the sum of 'h_value'
    total_support_value = df[metric].sum()

    # Normalize the 'h_value' column
    df[f'normalized_{metric}'] = df[metric] / total_support_value

    # Save the normalized data back to a new CSV file
    df.to_csv(path, index=False)
    return df


# def normalize_and_testing(pos_file, neg_file, top_n_lower, top_n_higher, orig_df, path):
#     pos_rules = pd.read_csv(pos_file)
#     neg_rules = pd.read_csv(neg_file)

#     for i in range(top_n_lower, top_n_higher + 1, 10):
#         print(f"Testing for top {i} rules.")
#         folder_path = f"{path}{i}/"
#         os.makedirs(folder_path, exist_ok=True)  # Ensure subfolder exists

#         pos_patterns_df = pos_rules.head(i).reset_index(drop=True)
#         neg_patterns_df = neg_rules.head(i).reset_index(drop=True)

#         pos_save = normalize_helper(
#             pos_patterns_df, f"{folder_path}pos_rules.csv")
#         neg_save = normalize_helper(
#             neg_patterns_df, f"{folder_path}neg_rules.csv")

#         cm(pos_save, neg_save, orig_df, f"{folder_path}cm.txt")

def normalize_and_testing(pos_file, neg_file, top_n_lower, top_n_higher, orig_df, path):
    pos_rules = pd.read_csv(pos_file)
    neg_rules = pd.read_csv(neg_file)
    
    # Define metrics to test
    #metrics = ['normalized_support', 'new_support', 'h_support', 'new_weights']
    metrics = ['normalized_support', 'coverage', 'h_value', 'ch_value','oh_value']
    # Create a summary dataframe to store all results
    all_results = pd.DataFrame(columns=['metric', 'num_rules', 'accuracy'])
    
    # Process each metric separately
    for metric in metrics:
        print(f"\n--- Testing with {metric} ---")
        
        # Create a directory for this metric's results
        metric_path = f"{path}{metric}/"
        os.makedirs(metric_path, exist_ok=True)
        
        # Lists to store rules and accuracies for this metric
        rules_counts = []
        accuracies = []
        
        # Process different numbers of rules
        for i in range(top_n_lower, top_n_higher + 1, 10):
            print(f"Testing for top {i} rules using {metric}.")
            folder_path = f"{metric_path}{i}/"
            os.makedirs(folder_path, exist_ok=True)
            
            # Sort by the current metric and take top i rules
            pos_patterns_df = pos_rules.sort_values(metric, ascending=False).head(i).reset_index(drop=True)
            neg_patterns_df = neg_rules.sort_values(metric, ascending=False).head(i).reset_index(drop=True)
            
            # Normalize and save based on the selected metric
            pos_save = normalize_helper(pos_patterns_df, f"{folder_path}pos_rules.csv", metric)
            neg_save = normalize_helper(neg_patterns_df, f"{folder_path}neg_rules.csv", metric)
            
            # Call cm function and capture accuracy
            result_df = cm(pos_save, neg_save, orig_df, f"{folder_path}cm.txt", metric)
            
            # Extract accuracy from the cm.txt file
            with open(f"{folder_path}cm.txt", 'r') as f:
                content = f.read()
                accuracy_line = [line for line in content.split('\n') if "Accuracy:" in line][0]
                accuracy = float(accuracy_line.split(': ')[1])
                
                # Store rule count and accuracy
                rules_counts.append(i)
                accuracies.append(accuracy)
                
                # Add to the summary dataframe
                all_results = pd.concat([all_results, pd.DataFrame({'metric': [metric], 
                                                                  'num_rules': [i], 
                                                                  'accuracy': [accuracy]})], 
                                       ignore_index=True)
                
                print(f"Accuracy for top {i} rules using {metric}: {accuracy}")
        
        # Create a plot for this metric
        plt.figure(figsize=(10, 6))
        plt.plot(rules_counts, accuracies, marker='o', linestyle='-', color='blue')
        plt.xlabel('Number of Top Rules')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy vs Number of Top Rules ({metric})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rules_counts)
        plt.ylim([min(accuracies) - 0.05, min(1.0, max(accuracies) + 0.05)])
        
        # Add accuracy values as labels above points
        for i, (x, y) in enumerate(zip(rules_counts, accuracies)):
            plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        plt.tight_layout()
        
        # Save the plot in the metric directory
        graph_path = f"{metric_path}accuracy_vs_rules_summary.png"
        plt.savefig(graph_path)
        print(f"Graph for {metric} saved to {graph_path}")
        
        # Save the accuracy data
        summary_data = pd.DataFrame({
            'num_rules': rules_counts,
            'accuracy': accuracies
        })
        summary_data.to_csv(f"{metric_path}accuracy_summary.csv", index=False)
    

normalize_and_testing(f'../Rules/{sys.argv[1]}/All_Weights/pos_rules.csv',f'../Rules/{sys.argv[1]}/All_Weights/neg_rules.csv',10,100,pd.read_csv(f'../Dataset/{sys.argv[1]}/Train_Test/{sys.argv[1]}_test.csv'),f'../Outputs/{sys.argv[1]}/')
