# sample.py
import torch
from src.models.bit_diffusion import Unet1D, BitDiffusion
import numpy as np
import os

# ==== Settings ====
output_file = "generated_sequences.txt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_samples = 10
seq_len = 30
cond_value = 0.5  # Target Delta_G_MFE in normalized scale [0, 1]

# ==== Initialize model ====
unet = Unet1D(dim=64, seq_len=seq_len, channels=4, cond_dim=1)
diff_model = BitDiffusion(unet, timesteps=1000).to(device)
diff_model.load_state_dict(torch.load("../src/checkpoint/best_model.pt", map_location=device))
diff_model.eval()

# ==== Generate samples ====
with torch.no_grad():
    cond_tensor = torch.full((num_samples, 1), cond_value).float().to(device)
    samples = diff_model.sample((num_samples, seq_len, 4), cond_tensor, device)

# ==== Decode one-hot to nucleotide sequences ====
vocab = ['A', 'C', 'G', 'U']
def decode_sequence(tensor):
    indices = torch.argmax(tensor, dim=-1).cpu().numpy()
    return ''.join([vocab[i] for i in indices])

decoded_sequences = [decode_sequence(seq) for seq in samples]

# ==== Save to file ====
with open(output_file, "w") as f:
    for i, seq in enumerate(decoded_sequences):
        f.write(f">seq_{i}\n{seq}\n")

print(f"✅ Generated {num_samples} sequences and saved to {output_file}")
