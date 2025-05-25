import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import logging
import fire
from model_utils import set_seed

def main(input_path='input/BankChurners.csv', output_dir='processed', seed=42):
    """
    Preprocesses the dataset: encodes categoricals, standardizes numericals, splits, and saves encoders/scaler.
    Also identifies and saves binary columns.
    """
    set_seed(seed)
    logging.basicConfig(level=logging.INFO)
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logging.error(f"Failed to read input file: {e}")
        return

    # Drop the last two columns
    df = df.iloc[:, :-2]

    # Handle missing values (simple fillna, could be improved)
    if df.isnull().any().any():
        logging.warning("Missing values found. Filling with column means/mode.")
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].mean())

    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    os.makedirs(output_dir, exist_ok=True)
    with open(f'{output_dir}/numerical_cols.pkl', 'wb') as f:
        pickle.dump(list(numerical_cols), f)

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Identify binary columns (after encoding)
    binary_cols = [col for col in df.columns if df[col].nunique() == 2]
    with open(f'{output_dir}/binary_cols.pkl', 'wb') as f:
        pickle.dump(binary_cols, f)

    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)

    train_df.to_parquet(f'{output_dir}/train.parquet', index=False)
    val_df.to_parquet(f'{output_dir}/val.parquet', index=False)
    test_df.to_parquet(f'{output_dir}/test.parquet', index=False)

    with open(f'{output_dir}/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    with open(f'{output_dir}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    logging.info('Preprocessing complete. Parquet files and encoders/scaler/binary_cols saved.')

if __name__ == '__main__':
    fire.Fire(main)
