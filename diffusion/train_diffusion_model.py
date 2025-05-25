import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Load the preprocessed train data
train_df = pd.read_parquet('processed/train.parquet')

# Convert DataFrame to tensor
data_tensor = torch.tensor(train_df.values, dtype=torch.float32)

print(data_tensor.shape)
print("Data tensor created")

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

# Define dataset and dataloader
class CreditCardDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = CreditCardDataset(data_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

print(dataloader)

# Initialize model, loss function, and optimizer
model = DiffusionModel(input_dim=data_tensor.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 10
for epoch in range(epochs):
    for batch in dataloader:
        # Add noise to the data
        noise = torch.randn_like(batch) * 0.1
        noisy_data = batch + noise

        # Forward pass
        output = model(noisy_data)

        # Compute loss
        loss = criterion(output, batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), 'processed/diffusion_model.pt')
print('Model saved to processed/diffusion_model.pt')
