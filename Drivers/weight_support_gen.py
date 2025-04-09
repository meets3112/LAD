import pandas as pd
import sys

# Load the datasets
heart_data = pd.read_csv(f'../Dataset/{sys.argv[1]}/Train_Test/{sys.argv[1]}_train.csv')
pos_rules_data = pd.read_csv(f'../Rules/{sys.argv[1]}/Original/pos_clean_rules.csv')
neg_rules_data = pd.read_csv(f'../Rules/{sys.argv[1]}/Original/neg_clean_rules.csv')

# Filter positive and negative instances
positive_instances = heart_data[heart_data['label'] == 1]
negative_instances = heart_data[heart_data['label'] == 0]
total_positive = len(positive_instances)
total_negative = len(negative_instances)

# Function to evaluate a rule
def evaluate_term(term, data_point):
    if "cp" in term:
        term = term.strip()
        is_negated = term.startswith("~")
        if is_negated:
            term = term[1:]  # Remove the negation symbol
        feature, condition = term.split('cp')
        feature = feature.strip()
        condition = condition.strip()
        if '>=' in condition:
            value = float(condition.split('>=')[1])
            return data_point[feature] >= value if not is_negated else data_point[feature] < value
        elif '<' in condition:
            value = float(condition.split('<')[1])
            return data_point[feature] < value if not is_negated else data_point[feature] >= value
    return False

# Function to calculate metrics for rules
def calculate_metrics(rules_data, instances, total_instances, label):
    coverage_values = []
    h_values = []
    ch_values = []

    for rule in rules_data['rule']:
        pattern = rule.split(' & ')
        covered_instances = instances[instances.apply(lambda row: all(evaluate_term(term, row) for term in pattern), axis=1)]
        num_covered = len(covered_instances)
        total_covered = len(heart_data[heart_data.apply(lambda row: all(evaluate_term(term, row) for term in pattern), axis=1)])

        # Calculate metrics
        coverage = num_covered / total_instances
        h_value = num_covered / total_covered if total_covered > 0 else 0
        ch_value = coverage * h_value

        coverage_values.append(coverage)
        h_values.append(h_value)
        ch_values.append(ch_value)

    # Add the new columns to the rules data
    rules_data[f'coverage'] = coverage_values
    rules_data[f'h_value'] = h_values
    rules_data[f'ch_value'] = ch_values

    return rules_data

# Calculate metrics for positive and negative rules
pos_rules_data = calculate_metrics(pos_rules_data, positive_instances, total_positive, 'pos')
neg_rules_data = calculate_metrics(neg_rules_data, negative_instances, total_negative, 'neg')
pos_rules_data['oh_value'] = pos_rules_data['normalized_support'] * pos_rules_data['h_value']
neg_rules_data['oh_value'] = neg_rules_data['normalized_support'] * neg_rules_data['h_value']



# Combine positive and negative rules into a single CSV

pos_rules_data.to_csv(f'../Rules/{sys.argv[1]}/All_Weights/pos_rules.csv', index=False)
neg_rules_data.to_csv(f'../Rules/{sys.argv[1]}/All_Weights/neg_rules.csv', index=False)

print("New CSV with coverage, h_value, oh_value and ch_value for positive and negative rules has been created")