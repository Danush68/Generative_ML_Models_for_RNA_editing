
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.models.vae_model import VAE

def loss_fn(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss / x.size(0)

# Load dataset
data = np.load('../data/processed_sequences.npy')  # Adapt path as needed
data_tensor = torch.tensor(data, dtype=torch.float32)
dataset = TensorDataset(data_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model setup
vae = VAE(input_dim=data.shape[1], latent_dim=32)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Training loop
vae.train()
epochs = 20
for epoch in range(epochs):
    total_loss = 0
    for batch in loader:
        x = batch[0]
        optimizer.zero_grad()
        recon, mu, logvar = vae(x)
        loss = loss_fn(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(loader):.4f}")

# Save model
torch.save(vae.state_dict(), '../outputs/vae_model.pth')

