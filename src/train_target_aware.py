# === train_target_aware.py (Fixed) ===

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from src.models.bit_diffusion import BitDiffusion, Unet1D

# === Config ===
BATCH_SIZE = 32
EPOCHS = 300
LR = 1e-4
TIMESTEPS = 1000
SEQ_LEN = 30
CHANNELS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "../data/processed"
SAVE_PATH = "../src/models/bit_diffusion_unet.pt"
PLOT_DIR = "../outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

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

def load_dataset(name):
    path = os.path.join(DATA_DIR, f"{name}.pt")
    data_dict = torch.load(path)
    dataset = GRNATensorDataset(data_dict)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(name == "train"))

train_loader = load_dataset("train")
val_loader = load_dataset("val")

# === Model and optimizer
unet = Unet1D(dim=256, seq_len=SEQ_LEN, channels=CHANNELS, cond_dim=1, target_dim=4)
model = BitDiffusion(unet, timesteps=TIMESTEPS, lambda_dg=1.0).to(DEVICE)
ema_model = deepcopy(model)
ema_decay = 0.999

def update_ema(ema_model, model):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_losses, val_losses = [], []

# === Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x = batch["x"].view(-1, SEQ_LEN, CHANNELS).to(DEVICE)
        cond_seq = batch["cond_seq"].to(DEVICE)
        cond_dg = batch["cond_dg"].to(DEVICE)

        t = torch.randint(0, TIMESTEPS, (x.size(0),), device=DEVICE).long()
        loss = model.p_losses(x, t, cond_dg, cond_seq, true_dg=cond_dg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema(ema_model, model)
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"\u2705 Epoch {epoch+1}: Avg Train Loss = {avg_train_loss:.6f}")

    # === Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            x = batch["x"].view(-1, SEQ_LEN, CHANNELS).to(DEVICE)
            cond_seq = batch["cond_seq"].to(DEVICE)
            cond_dg = batch["cond_dg"].to(DEVICE)

            t = torch.randint(0, TIMESTEPS, (x.size(0),), device=DEVICE).long()
            loss = model.p_losses(x, t, cond_dg, cond_seq, true_dg=cond_dg)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    print(f"\U0001F1F9\U0001F1ED Val Loss = {avg_val_loss:.6f}")
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    scheduler.step()

# Final model save
torch.save(model.state_dict(), SAVE_PATH)
print(f"\U0001F3AF Final model saved to {SAVE_PATH}")

# === Plot Loss Curves
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")
plt.grid(True)
plt.savefig(f"{PLOT_DIR}/loss_curve.png")
plt.show()