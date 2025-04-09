import pandas as pd


def replace_pos_operators(rule):
    rule = rule.replace('=', '>=')
    return rule

def replace_neg_operators(rule):
    if rule[0] == '!':
        rule = rule.replace('=', '<')
    rule.replace('!', '')
    return rule


def clean_rules(input_file, output_file):
    # Load the CSV file
    rules_df = pd.read_csv(input_file)

    rules_df['rule'] = rules_df['rule'].apply(replace_neg_operators)
    rules_df['rule'] = rules_df['rule'].apply(replace_pos_operators)

    def simplify_rule(rule):
        terms = rule.split(' & ')
        pos_simplified_terms = {}
        neg_simplified_terms = {}

        categorical_terms = set()  # Store categorical features separately

        # Process each term
        for term in terms:
            if '>=' in term:
                feature, condition = term.split('>=')
                feature = feature.strip()
                value = float(condition.strip())

                # Keep only the maximum condition for each feature
                if feature not in pos_simplified_terms or pos_simplified_terms[feature] < value:
                    pos_simplified_terms[feature] = value
            elif '<' in term:
                feature, condition = term.split('<')
                feature = feature.strip()
                value = float(condition.strip())

                # Keep only the minimum condition for each feature
                if feature not in neg_simplified_terms or neg_simplified_terms[feature] > value:
                    neg_simplified_terms[feature] = value
            else:
                # Categorical features (e.g., "default_no")
                categorical_terms.add(term.strip())

        # Reconstruct the rule from simplified terms
        numeric_part = ' & '.join(
                    [f"{feature} >= {value}" for feature, value in pos_simplified_terms.items()] +
                    [f"{feature} < {value}" for feature, value in neg_simplified_terms.items()]
                    )

        categorical_part = ' & '.join(categorical_terms)

        

        # Combine numeric and categorical conditions
        if numeric_part and categorical_part:
            simplified_rule = numeric_part + ' & ' + categorical_part
        elif numeric_part:
            simplified_rule = numeric_part
        else:
            simplified_rule = categorical_part

        return simplified_rule

    # Simplify each rule and create a new DataFrame with unique rules
    rules_df['simplified_rule'] = rules_df['rule'].apply(simplify_rule)

    # Sort by normalized_support in descending order to keep the highest value
    rules_df = rules_df.sort_values(by='normalized_support', ascending=False)

    # Drop duplicates based on simplified rules, keeping the highest support
    rules_df = rules_df.drop_duplicates(subset='simplified_rule')

    # Drop the original 'rule' column, rename 'simplified_rule' to 'rule'
    rules_df = rules_df.drop(columns='rule').rename(
        columns={'simplified_rule': 'rule'})

    # Reorder columns to 'rule' and 'normalized_support'
    rules_df = rules_df[['rule', 'normalized_support']]

    # Save the cleaned DataFrame to a new CSV file
    rules_df.to_csv(output_file, index=False)
    print(
        f"Cleaned rules saved to '{output_file}' with redundant conditions removed and duplicates deleted.")


# Apply the function to both positive and negative rules files
# clean_rules('output_files/neg_modified_rules.csv', 'Dataset/UNSW/neg_rules.csv')
# clean_rules('output_files/pos_modified_rules.csv', 'Dataset/UNSW/pos_rules.csv')
