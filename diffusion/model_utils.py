import torch
import torch.nn as nn
import random
import numpy as np

class DiffusionModel(nn.Module):
    """
    Autoencoder-like model for diffusion tasks, now accepts timestep embedding.
    """
    def __init__(self, input_dim, time_embed_dim=32):
        super(DiffusionModel, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, 256),
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

    def forward(self, x, t):
        # t: [batch] or [batch, 1]
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t_embed = self.time_embed(t.float())
        x = torch.cat([x, t_embed], dim=1)
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
