import pandas as pd
import torch
import torch.nn as nn
import pickle
import fire
import numpy as np

def main(num_records=None, from_noise=False):
    # Load encoders and scaler (if needed for inverse transform)
    with open('processed/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    with open('processed/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('processed/numerical_cols.pkl', 'rb') as f:
        true_numerical_cols = pickle.load(f)

    # Load the trained model definition
    class DiffusionModel(nn.Module):
        def __init__(self, input_dim):
            super(DiffusionModel, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim)
            )

        def forward(self, x):
            return self.net(x)

    # Determine input for inference
    if from_noise:
        # Use the shape of the test data to determine input_dim
        test_df = pd.read_parquet('processed/test.parquet')
        input_dim = test_df.shape[1]
        n_samples = int(num_records) if num_records is not None else len(test_df)
        # Generate random noise as input
        input_tensor = torch.randn((n_samples, input_dim), dtype=torch.float32)
        data_df = pd.DataFrame(np.zeros((n_samples, input_dim)), columns=test_df.columns)  # placeholder for column names
    else:
        # Use test data as input
        data_df = pd.read_parquet('processed/test.parquet')
        if num_records is not None:
            data_df = data_df.sample(n=int(num_records), random_state=42).reset_index(drop=True)
        input_tensor = torch.tensor(data_df.values, dtype=torch.float32)
        input_dim = input_tensor.shape[1]

    # Load the trained model
    model = DiffusionModel(input_dim=input_dim)
    model.load_state_dict(torch.load('processed/diffusion_model.pt'))
    model.eval()

    # Run inference
    with torch.no_grad():
        reconstructed = model(input_tensor).numpy()

    # Optionally, inverse transform numerical columns
    reconstructed_df = pd.DataFrame(reconstructed, columns=data_df.columns)
    reconstructed_df[true_numerical_cols] = scaler.inverse_transform(reconstructed_df[true_numerical_cols].values)

    # Save the reconstructed outputs
    mode = 'noise' if from_noise else 'test'
    out_path = f'processed/reconstructed_{mode}_{len(reconstructed_df)}.parquet'
    reconstructed_df.to_parquet(out_path, index=False)
    print(f'Inference complete. Reconstructed data saved to {out_path}')

if __name__ == '__main__':
    fire.Fire(main)
