import pandas as pd
import torch
import pickle
import logging
import json
import os
import fire
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model_utils import DiffusionModel, get_device, set_seed

def main(reference_path='processed/test.parquet', reconstructed_path=None, encoders_path='processed/label_encoders.pkl', scaler_path='processed/scaler.pkl', num_cols_path='processed/numerical_cols.pkl', device='cpu', seed=42, output_dir='processed'):
    """
    Compares a reference file (e.g., test data) with a reconstructed file (e.g., model output).
    Computes metrics and column-wise statistics.
    """
    set_seed(seed)
    logging.basicConfig(level=logging.INFO)
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(num_cols_path, 'rb') as f:
            true_numerical_cols = pickle.load(f)
        reference_df = pd.read_parquet(reference_path)
        if reconstructed_path is None:
            # Default to reconstructed_test_N.parquet if available
            files = [f for f in os.listdir(output_dir) if f.startswith('reconstructed_test_') and f.endswith('.parquet')]
            if not files:
                logging.error('No reconstructed file specified and none found in output_dir.')
                return
            reconstructed_path = os.path.join(output_dir, sorted(files)[-1])
        reconstructed_df = pd.read_parquet(reconstructed_path)
    except Exception as e:
        logging.error(f"Error loading files: {e}")
        return

    # Ensure columns match
    if list(reference_df.columns) != list(reconstructed_df.columns):
        logging.warning('Column order mismatch between reference and reconstructed data. Aligning columns.')
        reconstructed_df = reconstructed_df[reference_df.columns]

    # Compute metrics
    mse = mean_squared_error(reference_df, reconstructed_df)
    mae = mean_absolute_error(reference_df, reconstructed_df)
    r2 = r2_score(reference_df, reconstructed_df)

    logging.info(f"Evaluation Results: MSE: {mse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")

    # Save metrics
    metrics = {'MSE': mse, 'MAE': mae, 'R2': r2}
    with open(f'{output_dir}/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save column-wise statistics
    col_stats = {}
    for col in reference_df.columns:
        ref_mean = reference_df[col].mean()
        ref_std = reference_df[col].std()
        recon_mean = reconstructed_df[col].mean()
        recon_std = reconstructed_df[col].std()
        diff_mean = abs(ref_mean - recon_mean)
        diff_std = abs(ref_std - recon_std)
        flag = (diff_mean > 0.1 * abs(ref_mean)) or (diff_std > 0.1 * abs(ref_std))
        col_stats[col] = {
            'ref_mean': ref_mean,
            'ref_std': ref_std,
            'recon_mean': recon_mean,
            'recon_std': recon_std,
            'diff_mean': diff_mean,
            'diff_std': diff_std,
            'large_diff': flag
        }
    with open(f'{output_dir}/column_stats.json', 'w') as f:
        json.dump(col_stats, f, indent=2)

    # Save sample rows
    reference_df.head(5).to_csv(f'{output_dir}/sample_reference.csv', index=False)
    reconstructed_df.head(5).to_csv(f'{output_dir}/sample_reconstructed.csv', index=False)
    logging.info('Evaluation complete. Results saved.')

if __name__ == '__main__':
    fire.Fire(main)
