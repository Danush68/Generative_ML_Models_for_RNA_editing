
import torch
import numpy as np
from src.models.vae_model import VAE

# Load data
data = np.load("../../data/processed_sequences.npy")
data_tensor = torch.tensor(data, dtype=torch.float32)

# Load model
vae = VAE(input_dim=data.shape[1], latent_dim=32)
vae.load_state_dict(torch.load("../../outputs/vae_model.pth"))
vae.eval()

# Extract latent vectors
with torch.no_grad():
    latents = vae.encode(data_tensor).numpy()

# Save
np.save("../features/latent_vectors.npy", latents)
print("Saved latent vectors to features/latent_vectors.npy")
