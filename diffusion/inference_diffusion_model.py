import pandas as pd
import torch
import pickle
import numpy as np
import logging
import os
import fire
from model_utils import DiffusionModel, get_device, set_seed
from diffusion_utils import DiffusionProcess

def main(num_records=None, from_noise=False, test_path='processed/test.parquet', model_path='processed/diffusion_model.pt', encoders_path='processed/label_encoders.pkl', scaler_path='processed/scaler.pkl', num_cols_path='processed/numerical_cols.pkl', binary_cols_path='processed/binary_cols.pkl', train_path='processed/train.parquet', device='cpu', seed=42, output_dir='processed', diffusion_steps=1000):
    """
    Runs inference with the diffusion model on test data or random noise and saves the output.
    Post-processes binary columns to ensure they are 0 or 1, and standardizes numerical columns to match train data.
    If from_noise=True, uses DDPM-style generative sampling.
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
        with open(binary_cols_path, 'rb') as f:
            binary_cols = pickle.load(f)
        train_df = pd.read_parquet(train_path)
    except Exception as e:
        logging.error(f"Error loading encoders/scaler/binary_cols/train: {e}")
        return

    if from_noise:
        try:
            test_df = pd.read_parquet(test_path)
        except Exception as e:
            logging.error(f"Error loading test data for shape: {e}")
            return
        input_dim = test_df.shape[1]
        n_samples = int(num_records) if num_records is not None else len(test_df)
        diffusion = DiffusionProcess(timesteps=diffusion_steps, device=device)
        model = DiffusionModel(input_dim=input_dim).to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return
        model.eval()
        with torch.no_grad():
            generated = diffusion.p_sample_loop(model, (n_samples, input_dim)).cpu().numpy()
        reconstructed_df = pd.DataFrame(generated, columns=test_df.columns)
        data_df = pd.DataFrame(np.zeros((n_samples, input_dim)), columns=test_df.columns)
    else:
        try:
            data_df = pd.read_parquet(test_path)
        except Exception as e:
            logging.error(f"Error loading test data: {e}")
            return
        if num_records is not None:
            data_df = data_df.sample(n=int(num_records), random_state=seed).reset_index(drop=True)
        input_tensor = torch.tensor(data_df.values, dtype=torch.float32).to(device)
        input_dim = input_tensor.shape[1]
        model = DiffusionModel(input_dim=input_dim).to(device)
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return
        model.eval()
        with torch.no_grad():
            t = torch.zeros(input_tensor.size(0), dtype=torch.long, device=device)  # Use t=0 for all samples
            reconstructed = model(input_tensor, t).cpu().numpy()
        reconstructed_df = pd.DataFrame(reconstructed, columns=data_df.columns)

    reconstructed_df[true_numerical_cols] = scaler.inverse_transform(reconstructed_df[true_numerical_cols].values)

    # Post-process binary columns: clip to [0, 1] and round
    for col in binary_cols:
        reconstructed_df[col] = np.clip(reconstructed_df[col], 0, 1)
        reconstructed_df[col] = np.round(reconstructed_df[col]).astype(int)

    # Standardize numerical columns to match train data
    reconstructed_df[true_numerical_cols] = scaler.transform(reconstructed_df[true_numerical_cols].values)

    # Ensure column order matches train data
    reconstructed_df = reconstructed_df[train_df.columns]

    mode = 'noise' if from_noise else 'test'
    out_path = f'{output_dir}/reconstructed_{mode}_{len(reconstructed_df)}.parquet'
    reconstructed_df.to_parquet(out_path, index=False)
    logging.info(f'Inference complete. Reconstructed data saved to {out_path}')

if __name__ == '__main__':
    fire.Fire(main)
