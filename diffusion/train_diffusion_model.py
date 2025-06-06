import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import logging
import os
import json
import fire
from model_utils import DiffusionModel, get_device, set_seed
from diffusion_utils import DiffusionProcess

def main(train_path='processed/train.parquet', val_path='processed/val.parquet', output_dir='processed', batch_size=64, epochs=100, lr=1e-3, device='cpu', seed=42, patience=3, scheduler_factor=0.5, scheduler_patience=5, scheduler_min_lr=1e-6, diffusion_steps=1000):
    """
    Trains a DDPM-style diffusion model with validation and early stopping.
    Includes a ReduceLROnPlateau learning rate scheduler.
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
    diffusion = DiffusionProcess(timesteps=diffusion_steps, device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, min_lr=scheduler_min_lr, verbose=True
    )

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            t = diffusion.sample_timesteps(batch.size(0))
            noise = torch.randn_like(batch)
            x_noisy = diffusion.q_sample(batch, t, noise)
            pred_noise = model(x_noisy, t)
            loss = criterion(pred_noise, noise)
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
            t_val = diffusion.sample_timesteps(val_input.size(0))
            noise_val = torch.randn_like(val_input)
            x_noisy_val = diffusion.q_sample(val_input, t_val, noise_val)
            pred_noise_val = model(x_noisy_val, t_val)
            val_loss = criterion(pred_noise_val, noise_val).item()
        history['val_loss'].append(val_loss)

        # Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        logging.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

        # Step the scheduler
        scheduler.step(val_loss)

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
