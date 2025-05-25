import pandas as pd
import torch
import torch.nn as nn
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load encoders and scaler (if needed for inverse transform)
with open('processed/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)
with open('processed/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the list of true numerical columns
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

# Load test data
original_df = pd.read_parquet('processed/test.parquet')
input_tensor = torch.tensor(original_df.values, dtype=torch.float32)

# Load the trained model
model = DiffusionModel(input_dim=input_tensor.shape[1])
model.load_state_dict(torch.load('processed/diffusion_model.pt'))
model.eval()

# Run inference
with torch.no_grad():
    reconstructed = model(input_tensor).numpy()

# Optionally, inverse transform numerical columns
reconstructed_df = pd.DataFrame(reconstructed, columns=original_df.columns)
reconstructed_df[true_numerical_cols] = scaler.inverse_transform(reconstructed_df[true_numerical_cols].values)

# Inverse transform original numerical columns for fair comparison
original_df[true_numerical_cols] = scaler.inverse_transform(original_df[true_numerical_cols].values)

# Compute metrics
mse = mean_squared_error(original_df, reconstructed_df)
mae = mean_absolute_error(original_df, reconstructed_df)
r2 = r2_score(original_df, reconstructed_df)

print(f"Evaluation Results:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R^2: {r2:.4f}")

# Compare statistics for each column
print("\nColumn-wise comparison (mean and std):")
for col in original_df.columns:
    orig_mean = original_df[col].mean()
    orig_std = original_df[col].std()
    recon_mean = reconstructed_df[col].mean()
    recon_std = reconstructed_df[col].std()
    diff_mean = abs(orig_mean - recon_mean)
    diff_std = abs(orig_std - recon_std)
    flag = "<-- LARGE DIFF" if diff_mean > 0.1 * abs(orig_mean) or diff_std > 0.1 * abs(orig_std) else ""
    print(f"{col:25} | orig mean: {orig_mean:10.4f}, std: {orig_std:10.4f} | recon mean: {recon_mean:10.4f}, std: {recon_std:10.4f} {flag}")

# Show a few sample rows for visual inspection
print("\nSample original rows:")
print(original_df.head(5))
print("\nSample reconstructed rows:")
print(reconstructed_df.head(5))
