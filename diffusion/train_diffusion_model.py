import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import logging
import os
import json
import fire
from model_utils import DiffusionModel, get_device, set_seed

def main(train_path='processed/train.parquet', val_path='processed/val.parquet', output_dir='processed', batch_size=64, epochs=50, lr=1e-3, device='cpu', seed=42, patience=3):
    """
    Trains the diffusion model with validation and early stopping.
    """
    set_seed(seed)
    device = get_device(device)
    logging.basicConfig(level=logging.INFO)

    # Load data
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    data_tensor = torch.tensor(train_df.values, dtype=torch.float32)
    val_tensor = torch.tensor(val_df.values, dtype=torch.float32)

    class CreditCardDataset(Dataset):
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    dataset = CreditCardDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DiffusionModel(input_dim=data_tensor.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            noise = torch.randn_like(batch) * 0.1
            noisy_data = batch + noise
            output = model(noisy_data)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
        train_loss = running_loss / len(dataset)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_input = val_tensor.to(device)
            noise = torch.randn_like(val_input) * 0.1
            val_output = model(noise + val_input)
            val_loss = criterion(val_output, val_input).item()
        history['val_loss'].append(val_loss)

        logging.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{output_dir}/diffusion_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

    # Save training history
    with open(f'{output_dir}/training_history.json', 'w') as f:
        json.dump(history, f)
    logging.info('Training complete. Model and history saved.')

if __name__ == '__main__':
    fire.Fire(main)
