
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from copy import deepcopy

from notebooks.grna_dataset import GRNADataset  # (kept if you use elsewhere)
from src.models.bit_diffusion import BitDiffusion, Unet1D

# === Config ===
BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-4
TIMESTEPS = 1000
SEQ_LEN = 30
CHANNELS = 4
LAMBDA_DG_FIDELITY = 10.0

# --- mutation-count conditioning config ---
MUT_BIN_EDGES = [0, 2, 4, 6, 8, 10]  # bins: (0-2], (2-4], (4-6], (6-8], (8-10]
MUT_DIM = len(MUT_BIN_EDGES) - 1     # 5 one-hot bins
COND_DIM = 1 + MUT_DIM               # ΔG scalar + mut one-hot

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === .pt dataset paths
TRAIN_PT = "../data/processed/train.pt"
VAL_PT = "../data/processed/val.pt"

SAVE_PATH = "../src/models/bit_diffusion_unet.pt"
PLOT_DIR = "../outputs/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def bucketize_count(count: torch.Tensor):
    """
    count: (N,) int/float tensor of mutation counts
    returns: (N, MUT_DIM) one-hot over 5 bins
    """
    # bins: (0,2], (2,4], (4,6], (6,8], (8,10]
    bins = torch.bucketize(count.float(), torch.tensor(MUT_BIN_EDGES, dtype=torch.float32, device=count.device), right=True) - 1
    bins = bins.clamp(0, MUT_DIM-1)
    return torch.nn.functional.one_hot(bins.to(torch.long), num_classes=MUT_DIM).float()

# === Dataset Wrapper
class GRNAPTDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.x = data["x"]                # [N, L, C]
        self.cond_seq = data["cond_seq"]  # [N, L, 4]
        self.cond_dg = data["cond_dg"]    # [N, 1] (normalized)
        self.sample_weight = data.get("sample_weight", torch.ones(len(self.x)))
        # mutation info can be either mut_count or mut_bin (one-hot). Prefer explicit mut_bin if present.
        if "mut_bin" in data:
            self.mut_bin = data["mut_bin"].float()
        elif "mut_count" in data:
            mc = data["mut_count"].view(-1).to(torch.float32)
            self.mut_bin = bucketize_count(mc)
        else:
            raise KeyError("Dataset must include 'mut_count' or 'mut_bin' for mutation-count conditioning.")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return {
            "x": self.x[idx],
            "cond_seq": self.cond_seq[idx],
            "cond_dg": self.cond_dg[idx],
            "mut_bin": self.mut_bin[idx],
            "sample_weight": self.sample_weight[idx],
        }

# === Dataset & Dataloaders
train_dataset = GRNAPTDataset(TRAIN_PT)
val_dataset = GRNAPTDataset(VAL_PT)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# === Model and optimizer
unet = Unet1D(dim=256, seq_len=SEQ_LEN, channels=CHANNELS, cond_dim=COND_DIM, target_dim=4)
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
        # tiny noise to regularize dg
        noise = torch.empty_like(batch["cond_dg"]).uniform_(-0.01, 0.01)
        cond_dg = (batch["cond_dg"] + noise).clamp(0.0, 1.0).to(DEVICE)
        mut_bin = batch["mut_bin"].to(DEVICE)
        cond_vec = torch.cat([cond_dg, mut_bin], dim=-1)  # (B, 1+MUT_DIM)
        sample_weights = batch["sample_weight"].to(DEVICE)

        t = torch.randint(0, TIMESTEPS, (x.size(0),), device=DEVICE).long()
        raw_loss = model.p_losses(x, t, cond_vec, cond_seq, true_dg=cond_dg, reduction="none")
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
            mut_bin = batch["mut_bin"].to(DEVICE)
            cond_vec = torch.cat([cond_dg, mut_bin], dim=-1)
            sample_weights = batch["sample_weight"].to(DEVICE)

            t = torch.randint(0, TIMESTEPS, (x.size(0),), device=DEVICE).long()
            raw_loss = model.p_losses(x, t, cond_vec, cond_seq, true_dg=cond_dg, reduction="none")
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
