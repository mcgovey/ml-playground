import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os

# Load the dataset
input_path = 'input/BankChurners.csv'
df = pd.read_csv(input_path)

# Drop the last two columns
df = df.iloc[:, :-2]

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Save the list of true numerical columns before encoding
with open('processed/numerical_cols.pkl', 'wb') as f:
    pickle.dump(list(numerical_cols), f)

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Standardize numerical variables
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split into train, validation, and test sets
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Create output directory if it doesn't exist
os.makedirs('processed', exist_ok=True)

# Save splits as Parquet
train_df.to_parquet('processed/train.parquet', index=False)
val_df.to_parquet('processed/val.parquet', index=False)
test_df.to_parquet('processed/test.parquet', index=False)

# Save encoders and scaler
with open('processed/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)
with open('processed/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print('Preprocessing complete. Parquet files and encoders/scaler saved.')
