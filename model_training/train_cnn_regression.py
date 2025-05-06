
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import os

df = pd.read_csv("data/raw/hairpin_rna_random_mutations.csv")
df["Hairpin_RNA"] = df["Hairpin_RNA"].astype(str).str.replace(" ", "")
vocab = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
df["encoded_seq"] = df["Hairpin_RNA"].apply(lambda x: [vocab.get(c, 0) for c in x])
max_len = df["encoded_seq"].apply(len).max()
df["padded_seq"] = df["encoded_seq"].apply(lambda x: x + [0]*(max_len - len(x)))

X = torch.tensor(df["padded_seq"].tolist(), dtype=torch.long)
y = torch.tensor(df["Mutations"].tolist(), dtype=torch.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class RNADataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(RNADataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(RNADataset(X_test, y_test), batch_size=32)

class CNNRegressor(nn.Module):
    def __init__(self, vocab_size=4, embed_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(2)
        return self.fc(x).squeeze(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNRegressor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

train_losses, val_r2s = [], []

for epoch in range(5):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().tolist()
            all_preds += preds
            all_true += y_batch.tolist()
    r2 = r2_score(all_true, all_preds)
    val_r2s.append(r2)
    print(f"Epoch {epoch+1}: Loss = {train_losses[-1]:.4f}, R² = {r2:.4f}")

os.makedirs("outputs/models", exist_ok=True)
torch.save(model.state_dict(), "outputs/models/cnn_regressor.pth")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train MSE Loss")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_r2s, label="Validation R² Score", color="green")
plt.title("Validation R² Score over Epochs")
plt.xlabel("Epoch")
plt.ylabel("R²")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

rmse = np.sqrt(mean_squared_error(all_true, all_preds))
print(f"\n✅ Final RMSE: {rmse:.2f}, R² Score: {r2_score(all_true, all_preds):.2f}")
