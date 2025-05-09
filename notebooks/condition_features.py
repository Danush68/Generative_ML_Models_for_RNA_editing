import torch
import pandas as pd

# Load dataset
df = pd.read_csv("../data/processed/vienna_rna_full_features.csv")

# Extract and normalize Delta_G_MFE
delta_g = df["Delta_G_MFE"].astype(float)
delta_g_norm = (delta_g - delta_g.min()) / (delta_g.max() - delta_g.min())

# Convert to tensor [N, 1]
conditioning_tensor = torch.tensor(delta_g_norm.values).unsqueeze(1).float()

# Save
torch.save(conditioning_tensor, "../data/processed/delta_g_mfe.pt")
print(f"✅ Saved normalized Delta_G_MFE tensor of shape {conditioning_tensor.shape} to delta_g_mfe.pt")
