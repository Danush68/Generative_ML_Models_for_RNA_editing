
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv("data/raw/hairpin_rna_random_mutations.csv").head(1000).copy()
df["Hairpin_RNA"] = df["Hairpin_RNA"].astype(str).str.replace(" ", "", regex=False)
df["Secondary_Structure"] = ["." * len(seq) for seq in df["Hairpin_RNA"]]
df["label"] = (df["Mutations"] >= 5).astype(int)

vocab = {'A': 0, 'U': 1, 'G': 2, 'C': 3, '.': 4, '(': 5, ')': 6}
def encode(seq): return [vocab.get(char, 7) for char in seq]

max_len = max(df["Hairpin_RNA"].apply(len))
df["encoded_seq"] = df["Hairpin_RNA"].apply(lambda x: encode(x) + [0]*(max_len - len(x)))
df["encoded_struct"] = df["Secondary_Structure"].apply(lambda x: encode(x) + [0]*(max_len - len(x)))

X_seq = torch.tensor(df["encoded_seq"].tolist(), dtype=torch.long)
X_struct = torch.tensor(df["encoded_struct"].tolist(), dtype=torch.long)
y = torch.tensor(df["label"].tolist(), dtype=torch.long)

X_seq_train, X_seq_test, X_struct_train, X_struct_test, y_train, y_test = train_test_split(
    X_seq, X_struct, y, test_size=0.2, random_state=42
)

class HairpinDataset(Dataset):
    def __init__(self, seqs, structs, labels):
        self.seqs = seqs
        self.structs = structs
        self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.seqs[idx], self.structs[idx], self.labels[idx]

train_loader = DataLoader(HairpinDataset(X_seq_train, X_struct_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(HairpinDataset(X_seq_test, X_struct_test, y_test), batch_size=32)

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size=8, embed_dim=16, num_classes=2):
        super().__init__()
        self.embed_seq = nn.Embedding(vocab_size, embed_dim)
        self.embed_struct = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(2 * embed_dim, 64, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, seq, struct):
        seq_emb = self.embed_seq(seq).permute(0, 2, 1)
        struct_emb = self.embed_struct(struct).permute(0, 2, 1)
        x = torch.cat([seq_emb, struct_emb], dim=1)
        x = self.pool(torch.relu(self.conv(x))).squeeze(2)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_losses, val_accuracies = [], []

for epoch in range(5):
    model.train()
    total_loss = 0
    for seqs, structs, labels in train_loader:
        seqs, structs, labels = seqs.to(device), structs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(seqs, structs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, structs, labels in test_loader:
            seqs, structs = seqs.to(device), structs.to(device)
            preds = torch.argmax(model(seqs, structs), dim=1).cpu().tolist()
            all_preds += preds
            all_labels += labels.tolist()
    acc = accuracy_score(all_labels, all_preds)
    val_accuracies.append(acc)
    print(f"Epoch {epoch+1}: Loss = {train_losses[-1]:.4f}, Accuracy = {acc:.4f}")

os.makedirs("outputs/models", exist_ok=True)
torch.save(model.state_dict(), "outputs/models/cnn_mutation.pth")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Validation Accuracy", color="orange")
plt.title("Validation Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
