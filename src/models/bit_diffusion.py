import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None].float() * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    return emb

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, timesteps, steps)
    alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 1e-7, 1e-2)

class Unet1D(nn.Module):
    def __init__(self, dim=256, seq_len=30, channels=4, cond_dim=1, target_dim=4, time_emb_dim=128):
        super().__init__()
        self.seq_len = seq_len

        self.input_proj = nn.Conv1d(channels + target_dim, dim, kernel_size=3, padding=1)
        self.down1 = nn.Conv1d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv1d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1)

        self.film1 = nn.Linear(cond_dim, dim * 2)
        self.film2 = nn.Linear(cond_dim, dim * 4)
        self.film3 = nn.Linear(cond_dim, dim * 8)

        self.time_proj1 = nn.Linear(time_emb_dim, dim)
        self.time_proj2 = nn.Linear(time_emb_dim, dim * 2)
        self.time_proj3 = nn.Linear(time_emb_dim, dim * 4)

        self.up1 = nn.ConvTranspose1d(dim * 4, dim * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose1d(dim * 2, dim, kernel_size=4, stride=2, padding=1)
        self.output_proj = nn.Conv1d(dim, channels, kernel_size=1)

    def forward(self, gRNA, target_rna, cond, timestep):
        x = torch.cat([gRNA, target_rna], dim=-1).permute(0, 2, 1)
        t_emb = get_timestep_embedding(timestep, self.time_proj1.in_features)

        scale1, shift1 = self.film1(cond).chunk(2, dim=-1)
        scale2, shift2 = self.film2(cond).chunk(2, dim=-1)
        scale3, shift3 = self.film3(cond).chunk(2, dim=-1)

        t1 = self.time_proj1(t_emb).unsqueeze(-1)
        t2 = self.time_proj2(t_emb).unsqueeze(-1)
        t3 = self.time_proj3(t_emb).unsqueeze(-1)

        x1 = F.relu(self.input_proj(x) * (1 + scale1.unsqueeze(-1)) + shift1.unsqueeze(-1) + t1)
        x2 = F.relu(self.down1(x1) * (1 + scale2.unsqueeze(-1)) + shift2.unsqueeze(-1) + t2)
        x3 = F.relu(self.down2(x2) * (1 + scale3.unsqueeze(-1)) + shift3.unsqueeze(-1) + t3)

        u1 = F.relu(self.up1(x3))[:, :, :x2.shape[2]] + x2
        u2 = F.relu(self.up2(u1))[:, :, :x1.shape[2]] + x1

        out = self.output_proj(u2)
        return out.permute(0, 2, 1)

class BitDiffusion(nn.Module):
    def __init__(self, model: nn.Module, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        betas = cosine_beta_schedule(timesteps)
        self.register_buffer("betas", torch.tensor(betas, dtype=torch.float32))
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - self.betas, dim=0))

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        alpha_t = self.alphas_cumprod.to(t.device)[t].view(-1, 1, 1)
        return alpha_t.sqrt() * x_start + (1 - alpha_t).sqrt() * noise

    def p_losses(self, x_start, t, cond, target):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        pred_noise = self.model(x_noisy, target, cond, t)
        return F.mse_loss(pred_noise, noise)

    def sample(self, shape, cond, target, device):
        x = torch.randn(shape).to(device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=device)
            pred_noise = self.model(x, target, cond, t_tensor)
            alpha_t = self.alphas.to(device)[t]
            alpha_cumprod_t = self.alphas_cumprod.to(device)[t]
            mean = (x - (1 - alpha_t).sqrt() * pred_noise) / alpha_t.sqrt()
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + (1 - alpha_cumprod_t).sqrt() * noise
            else:
                x = mean
        return x
