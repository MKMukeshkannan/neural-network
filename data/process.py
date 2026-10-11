'''
!!WARN!! THIS FILE IS COMPLETELY AI GENERATED
this one reads the data.csv file, -> normalizes it and splits it into train and test set (90% and 10%)
'''
import pandas as pd

# Input CSV
input_file = "data.csv"

# Output files
train_file = "train.csv"
test_file = "test.csv"

# Read CSV
df = pd.read_csv(input_file)

# Normalize numeric columns using min-max scaling
numeric_columns = df.select_dtypes(include="number").columns

for column in numeric_columns:
    min_value = df[column].min()
    max_value = df[column].max()

    if max_value != min_value:
        df[column] = (df[column] - min_value) / (max_value - min_value)

# Normalize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

# 90-10 train-test split
train_df = df.sample(frac=0.9, random_state=42)
test_df = df.drop(train_df.index)

# Save to separate CSV files
train_df.to_csv(train_file, index=False)
test_df.to_csv(test_file, index=False)

print(f"Train set: {len(train_df)} rows -> {train_file}")
print(f"Test set:  {len(test_df)} rows -> {test_file}")
