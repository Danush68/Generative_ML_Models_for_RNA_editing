# prepare_target_onehot.py

import torch
import os

# Input & output paths
input_path = "../data/processed/train.pt"
output_path = "../data/processed/target_onehot.pt"

# Load full dataset
data = torch.load(input_path)

# Extract only the target RNA (cond_seq)
target_onehot = data["cond_seq"]

# Save to a lightweight .pt file
torch.save(target_onehot, output_path)
print(f"✅ Saved target_onehot.pt with shape {target_onehot.shape} to '{output_path}'")
