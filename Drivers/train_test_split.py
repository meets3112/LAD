import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# Load the dataset
data = pd.read_csv(f'../Dataset/{sys.argv[1]}/Original/{sys.argv[1]}.csv')

# Split the dataset into train (80%) and test (20%) sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save the train and test datasets to separate CSV files
train_data.to_csv(f'../Dataset/{sys.argv[1]}/Train_Test/{sys.argv[1]}_train.csv', index=False)
test_data.to_csv(f'../Dataset/{sys.argv[1]}/Train_Test/{sys.argv[1]}_test.csv', index=False)

print("Train and test datasets have been created:")
print(f"Train dataset size: {len(train_data)}")
print(f"Test dataset size: {len(test_data)}")