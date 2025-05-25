import torch
import torch.nn as nn
import random
import numpy as np

class DiffusionModel(nn.Module):
    """
    More complex autoencoder-like model for diffusion tasks.
    Now includes more layers, batch normalization, and dropout.
    """
    def __init__(self, input_dim):
        super(DiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        return self.net(x)

def get_device(device_preference=None):
    """
    Returns the appropriate torch device (cuda if available and requested, else cpu).
    """
    if device_preference == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def set_seed(seed=42):
    """
    Sets random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
