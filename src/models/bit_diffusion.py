import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet1D(nn.Module):
    def __init__(self, dim=64, seq_len=30, channels=4, cond_dim=1, target_dim=4):
        super().__init__()
        self.seq_len = seq_len
        self.input_proj = nn.Conv1d(channels + target_dim, dim, kernel_size=3, padding=1)  # gRNA + target RNA

        self.down1 = nn.Conv1d(dim, dim * 2, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv1d(dim * 2, dim * 4, kernel_size=3, stride=2, padding=1)

        self.cond_proj1 = nn.Linear(cond_dim, dim)
        self.cond_proj2 = nn.Linear(cond_dim, dim * 2)
        self.cond_proj3 = nn.Linear(cond_dim, dim * 4)

        self.up1 = nn.ConvTranspose1d(dim * 4, dim * 2, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose1d(dim * 2, dim, kernel_size=4, stride=2, padding=1)
        self.output_proj = nn.Conv1d(dim, channels, kernel_size=1)

    def forward(self, gRNA, target_rna, cond):
        x = torch.cat([gRNA, target_rna], dim=-1).permute(0, 2, 1)  # [B, C, L]

        x1 = F.relu(self.input_proj(x))
        x1 = x1 + self.cond_proj1(cond).unsqueeze(-1)

        x2 = F.relu(self.down1(x1))
        x2 = x2 + self.cond_proj2(cond).unsqueeze(-1)

        x3 = F.relu(self.down2(x2))
        x3 = x3 + self.cond_proj3(cond).unsqueeze(-1)

        u1 = F.relu(self.up1(x3))
        u1 = u1[:, :, :x2.shape[2]] + x2

        u2 = F.relu(self.up2(u1))
        u2 = u2[:, :, :x1.shape[2]] + x1

        out = self.output_proj(u2)
        return out.permute(0, 2, 1)  # [B, L, C]

class BitDiffusion(nn.Module):
    def __init__(self, model: nn.Module, timesteps=1000):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod = self.alphas_cumprod[t].sqrt().view(-1, 1, 1)
        sqrt_one_minus = (1 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1)
        return sqrt_alpha_cumprod * x_start + sqrt_one_minus * noise

    def p_losses(self, x_start, t, cond, target):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted = self.model(x_noisy, target, cond)
        return F.mse_loss(predicted, noise)

    def sample(self, shape, cond, target, device):
        x = torch.randn(shape).to(device)
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((shape[0],), t, dtype=torch.long).to(device)
            predicted_noise = self.model(x, target, cond)
            alpha_t = self.alphas[t].to(device)
            alpha_cumprod_t = self.alphas_cumprod[t].to(device)

            mean = (x - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
            if t > 0:
                noise = torch.randn_like(x)
                x = mean + (1 - alpha_cumprod_t).sqrt() * noise
            else:
                x = mean
        return x
