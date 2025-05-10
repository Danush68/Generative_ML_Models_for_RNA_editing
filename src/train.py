# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.models.bit_diffusion import Unet1D, BitDiffusion
import os
import csv
from torch.utils.tensorboard import SummaryWriter

# ==== Settings ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
epochs = 30
lr = 1e-4

# ==== Load Data ====
data_dir = "../data/processed"
x = torch.load(os.path.join(data_dir, "mutated_gRNA_onehot.pt"))  # [N, 30, 4]
cond = torch.load(os.path.join(data_dir, "delta_g_mfe.pt"))        # [N, 1]

dataset = TensorDataset(x, cond)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ==== Initialize Model ====
unet = Unet1D(dim=64, seq_len=30, channels=4, cond_dim=1)
diff_model = BitDiffusion(unet, timesteps=1000).to(device)
optimizer = torch.optim.Adam(diff_model.parameters(), lr=lr)

# ==== Setup Logging ====
loss_log = []
writer = SummaryWriter(log_dir="runs/bit_diff")
best_loss = float('inf')

# ==== Training Loop ====
for epoch in range(1, epochs + 1):
    diff_model.train()
    total_loss = 0
    for batch_x, batch_cond in dataloader:
        batch_x, batch_cond = batch_x.to(device), batch_cond.to(device)
        t = torch.randint(0, diff_model.timesteps, (batch_x.size(0),), device=device).long()

        loss = diff_model.p_losses(batch_x, t, batch_cond)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    avg_loss = total_loss / len(dataset)
    loss_log.append((epoch, avg_loss))
    writer.add_scalar("Loss/train", avg_loss, epoch)

    print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.6f}")

    # Save checkpoint
    torch.save(diff_model.state_dict(), f"../src/checkpoint/checkpoint_epoch_{epoch}.pt")

    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(diff_model.state_dict(), "../src/checkpoint/best_model.pt")

# ==== Save Loss Log ====
with open("training_loss_log.csv", "w", newline="") as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow(["Epoch", "Loss"])
    writer_csv.writerows(loss_log)

writer.close()
print("✅ Training complete.")
