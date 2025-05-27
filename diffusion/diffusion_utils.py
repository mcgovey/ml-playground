import torch
import numpy as np

class DiffusionProcess:
    """
    Implements the forward (q_sample) and reverse (denoising) processes for DDPM on tabular data.
    Uses a linear noise schedule by default.
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alpha_cumprod[:-1]], dim=0)

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (add noise) at timestep t
        x_start: [batch, features]
        t: [batch] (timesteps)
        noise: [batch, features] (optional)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod = self.alpha_cumprod[t].sqrt().unsqueeze(-1)
        sqrt_one_minus_alpha_cumprod = (1 - self.alpha_cumprod[t]).sqrt().unsqueeze(-1)
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus_alpha_cumprod * noise

    def sample_timesteps(self, batch_size):
        return torch.randint(0, self.timesteps, (batch_size,), device=self.device)

    def get_variance(self, t):
        return self.betas[t]

    def p_sample_loop(self, model, shape):
        """
        DDPM-style generative sampling: iteratively denoise from pure noise.
        model: the trained diffusion model
        shape: (batch, features)
        Returns: generated samples [batch, features]
        """
        device = self.device
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            with torch.no_grad():
                pred_noise = model(x, t_batch)
            alpha = self.alphas[t]
            alpha_cumprod = self.alpha_cumprod[t]
            beta = self.betas[t]
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = (1 / alpha.sqrt()) * (x - (beta / (1 - alpha_cumprod).sqrt()) * pred_noise) + beta.sqrt() * noise
        return x
