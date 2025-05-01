import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load and preprocess dataset
# -----------------------------
df = pd.read_csv("../../data/raw/hairpin_rna_random_mutations.csv")
df["Hairpin_RNA"] = df["Hairpin_RNA"].astype(str).str.replace(" ", "")
df["label"] = (df["Mutations"] >= 5).astype(int)

# Encode RNA sequence
vocab = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
def encode(seq):
    return [vocab.get(base, 0) for base in seq]

max_len = df["Hairpin_RNA"].str.len().max()
df["encoded_seq"] = df["Hairpin_RNA"].apply(lambda x: encode(x) + [0]*(max_len - len(x)))

X = torch.tensor(df["encoded_seq"].tolist(), dtype=torch.long)
y = torch.tensor(df["label"].tolist(), dtype=torch.long)

# -----------------------------
# 2. Train-test split and Dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

class RNASequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

train_loader = DataLoader(RNASequenceDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(RNASequenceDataset(X_test, y_test), batch_size=32)

# -----------------------------
# 3. CNN Model
# -----------------------------
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size=4, embed_dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(2)
        return self.fc(x)

# -----------------------------
# 4. Training
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_losses, val_accuracies = [], []

for epoch in range(5):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    # Validation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = torch.argmax(model(X_batch), dim=1).cpu().tolist()
            all_preds += preds
            all_labels += y_batch.tolist()
    acc = accuracy_score(all_labels, all_preds)
    val_accuracies.append(acc)
    print(f"Epoch {epoch+1}: Loss = {train_losses[-1]:.4f}, Accuracy = {acc:.4f}")

# -----------------------------
# 5. Plot Training Results
# -----------------------------
plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Validation Accuracy", color="orange")
plt.title("Validation Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

