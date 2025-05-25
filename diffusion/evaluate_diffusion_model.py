import pandas as pd
import torch
import pickle
import logging
import json
import os
import fire
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model_utils import DiffusionModel, get_device, set_seed

def main(test_path='processed/test.parquet', model_path='processed/diffusion_model.pt', encoders_path='processed/label_encoders.pkl', scaler_path='processed/scaler.pkl', num_cols_path='processed/numerical_cols.pkl', device='cpu', seed=42, output_dir='processed'):
    """
    Evaluates the diffusion model on test data and saves metrics/statistics.
    """
    set_seed(seed)
    device = get_device(device)
    logging.basicConfig(level=logging.INFO)
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(num_cols_path, 'rb') as f:
            true_numerical_cols = pickle.load(f)
        original_df = pd.read_parquet(test_path)
    except Exception as e:
        logging.error(f"Error loading files: {e}")
        return

    input_tensor = torch.tensor(original_df.values, dtype=torch.float32).to(device)
    model = DiffusionModel(input_dim=input_tensor.shape[1]).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return
    model.eval()

    with torch.no_grad():
        reconstructed = model(input_tensor).cpu().numpy()

    reconstructed_df = pd.DataFrame(reconstructed, columns=original_df.columns)
    reconstructed_df[true_numerical_cols] = scaler.inverse_transform(reconstructed_df[true_numerical_cols].values)
    original_df[true_numerical_cols] = scaler.inverse_transform(original_df[true_numerical_cols].values)

    mse = mean_squared_error(original_df, reconstructed_df)
    mae = mean_absolute_error(original_df, reconstructed_df)
    r2 = r2_score(original_df, reconstructed_df)

    logging.info(f"Evaluation Results: MSE: {mse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")

    # Save metrics
    metrics = {'MSE': mse, 'MAE': mae, 'R2': r2}
    with open(f'{output_dir}/evaluation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save column-wise statistics
    col_stats = {}
    for col in original_df.columns:
        orig_mean = original_df[col].mean()
        orig_std = original_df[col].std()
        recon_mean = reconstructed_df[col].mean()
        recon_std = reconstructed_df[col].std()
        diff_mean = abs(orig_mean - recon_mean)
        diff_std = abs(orig_std - recon_std)
        flag = (diff_mean > 0.1 * abs(orig_mean)) or (diff_std > 0.1 * abs(orig_std))
        col_stats[col] = {
            'orig_mean': orig_mean,
            'orig_std': orig_std,
            'recon_mean': recon_mean,
            'recon_std': recon_std,
            'diff_mean': diff_mean,
            'diff_std': diff_std,
            'large_diff': flag
        }
    with open(f'{output_dir}/column_stats.json', 'w') as f:
        json.dump(col_stats, f, indent=2)

    # Save sample rows
    original_df.head(5).to_csv(f'{output_dir}/sample_original.csv', index=False)
    reconstructed_df.head(5).to_csv(f'{output_dir}/sample_reconstructed.csv', index=False)
    logging.info('Evaluation complete. Results saved.')

if __name__ == '__main__':
    fire.Fire(main)
