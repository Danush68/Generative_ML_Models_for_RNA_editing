import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from copy import deepcopy

from notebooks.grna_dataset import GRNADataset
from src.models.bit_diffusion import BitDiffusion, Unet1D

# === Config ===
BATCH_SIZE = 128
EPOCHS = 300
LR = 1e-4
TIMESTEPS = 1000
SEQ_LEN = 30
CHANNELS = 2
LAMBDA_DG_FIDELITY = 10.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === .pt dataset paths
TRAIN_PT = "../data/processed/train.pt"
VAL_PT = "../data/processed/val.pt"

SAVE_PATH = "../src/models/bit_diffusion_unet.pt"
PLOT_DIR = "../outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# === Dataset Wrapper
class GRNAPTDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.x = data["x"]                # shape: [N, seq_len, channels]
        self.cond_seq = data["cond_seq"]  # shape: [N, seq_len, 4]
        self.cond_dg = data["cond_dg"]    # shape: [N, 1]
        self.sample_weight = data["sample_weight"]  # shape: [N]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "cond_seq": self.cond_seq[idx],
            "cond_dg": self.cond_dg[idx],
            "sample_weight": self.sample_weight[idx],
        }

# === Dataset & Dataloaders
train_dataset = GRNAPTDataset(TRAIN_PT)
val_dataset = GRNAPTDataset(VAL_PT)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Model and optimizer
unet = Unet1D(dim=256, seq_len=SEQ_LEN, channels=CHANNELS, cond_dim=1, target_dim=4)
model = BitDiffusion(unet, timesteps=TIMESTEPS, lambda_dg=LAMBDA_DG_FIDELITY).to(DEVICE)
ema_model = deepcopy(model)
ema_decay = 0.999

def update_ema(ema_model, model):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

train_losses = []
val_losses = []

# === Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        x = batch["x"].view(-1, SEQ_LEN, CHANNELS).to(DEVICE)
        cond_seq = batch["cond_seq"].to(DEVICE)
        noise = torch.empty_like(batch["cond_dg"]).uniform_(-0.01, 0.01)
        cond_dg = (batch["cond_dg"] + noise).clamp(0.0, 1.0).to(DEVICE)
        sample_weights = batch["sample_weight"].to(DEVICE)

        t = torch.randint(0, TIMESTEPS, (x.size(0),), device=DEVICE).long()
        raw_loss = model.p_losses(x, t, cond_dg, cond_seq, true_dg=cond_dg, reduction="none")
        loss = (raw_loss * sample_weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        update_ema(ema_model, model)
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # === Validation
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
            x = batch["x"].view(-1, SEQ_LEN, CHANNELS).to(DEVICE)
            cond_seq = batch["cond_seq"].to(DEVICE)
            cond_dg = batch["cond_dg"].to(DEVICE)
            sample_weights = batch["sample_weight"].to(DEVICE)

            t = torch.randint(0, TIMESTEPS, (x.size(0),), device=DEVICE).long()
            raw_loss = model.p_losses(x, t, cond_dg, cond_seq, true_dg=cond_dg, reduction="none")
            val_loss = (raw_loss * sample_weights).mean()
            total_val_loss += val_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    scheduler.step()
    print(f"✅ Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

# === Save final model
torch.save(model.state_dict(), SAVE_PATH)
print(f"🎯 Final model saved to {SAVE_PATH}")

# === Plot training & validation loss
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training vs Validation Loss")
plt.grid(True)
plt.savefig(f"{PLOT_DIR}/loss_curve.png")
plt.show()
