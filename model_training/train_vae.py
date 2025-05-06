
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
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
train_losses = []

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
    avg_loss = total_loss / len(loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Save model
os.makedirs("../outputs/models", exist_ok=True)
torch.save(vae.state_dict(), '../outputs/models/vae_model.pth')

# Plot training loss
#os.makedirs("../outputs/models", exist_ok=True)
plt.figure()
plt.plot(range(1, epochs + 1), train_losses, marker='o')
plt.title("VAE Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("../outputs/plots/vae_loss_curve.png")
plt.show()

print(f"✅ Final Loss: {train_losses[-1]:.4f}")
