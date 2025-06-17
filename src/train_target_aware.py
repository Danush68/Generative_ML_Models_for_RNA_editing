# train_target_aware.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

from src.models.bit_diffusion import BitDiffusion, Unet1D

# === Config ===
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-4
TIMESTEPS = 1000
SEQ_LEN = 30
CHANNELS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "../data/processed"
SAVE_PATH = "../src/models/bit_diffusion_unet.pt"

# === Dataset Wrapper ===
class GRNATensorDataset(Dataset):
    def __init__(self, data_dict):
        self.x = data_dict["x"]
        self.cond_seq = data_dict["cond_seq"]
        self.cond_dg = data_dict["cond_dg"]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "cond_seq": self.cond_seq[idx],
            "cond_dg": self.cond_dg[idx]
        }

# === Load .pt datasets ===
def load_dataset(name):
    path = os.path.join(DATA_DIR, f"{name}.pt")
    data_dict = torch.load(path)
    dataset = GRNATensorDataset(data_dict)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(name == "train"))

train_loader = load_dataset("train")
val_loader = load_dataset("val")

# === Model ===
unet = Unet1D(dim=128, seq_len=SEQ_LEN, channels=CHANNELS, cond_dim=1, target_dim=4)
model = BitDiffusion(unet, timesteps=TIMESTEPS).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# === Training Loop ===
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x = batch["x"].view(-1, SEQ_LEN, CHANNELS).to(DEVICE)
        cond_seq = batch["cond_seq"].to(DEVICE)
        cond_dg = batch["cond_dg"].to(DEVICE)

        t = torch.randint(0, TIMESTEPS, (x.size(0),), device=DEVICE).long()
        loss = model.p_losses(x, t, cond_dg, cond_seq)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.6f}")

    # === Validation ===
    model.eval()
    with torch.no_grad():
        total_val_loss = 0.0
        for batch in val_loader:
            x = batch["x"].view(-1, SEQ_LEN, CHANNELS).to(DEVICE)
            cond_seq = batch["cond_seq"].to(DEVICE)
            cond_dg = batch["cond_dg"].to(DEVICE)

            t = torch.randint(0, TIMESTEPS, (x.size(0),), device=DEVICE).long()
            loss = model.p_losses(x, t, cond_dg, cond_seq)
            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"🧪 Validation Loss = {avg_val_loss:.6f}")

# === Save model ===
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), SAVE_PATH)
print(f"📦 Model saved to {SAVE_PATH}")
