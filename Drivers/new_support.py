import pandas as pd

# Load the datasets
heart_data = pd.read_csv('Dataset/Heart/heart_train.csv')
rules_data = pd.read_csv('Dataset/H_value/neg_clean_rules_h.csv')

# Filter positive instances
instances = heart_data
total_positive = len(instances)
positive_instances = instances[instances['label'] == 1]
negative_instances = instances[instances['label'] == 0]

# Function to evaluate a rule
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

# Calculate new support values
new_support_values = []
for rule in rules_data['rule']:
    pattern = rule.split(' & ')
    pos_support_value = 0
    neg_support_value = 0
    for _, row in positive_instances.iterrows():
        if all(evaluate_term(term, row) for term in pattern):
            pos_support_value+=1
    for _, row in negative_instances.iterrows():
        if all(evaluate_term(term, row) for term in pattern):
            neg_support_value+=1
    new_support_values.append((neg_support_value)/(pos_support_value+neg_support_value))
    

# Add the new column to the rules data
rules_data['h_support'] = new_support_values
#normalize new supports between 0 and 1
rules_data['ch_support'] = rules_data['new_support'] * rules_data['h_support']
rules_data['normalized_ch_support'] = rules_data['ch_support'] 
# Save the updated CSV
rules_data.to_csv('Dataset/H_value/neg_clean_rules_ch.csv', index=False)