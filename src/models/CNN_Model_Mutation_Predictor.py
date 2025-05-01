import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import random
import matplotlib.pyplot as plt

# Simulate enhanced dataset (with mock secondary structure)
df = pd.read_csv("../../data/raw/hairpin_rna_random_mutations.csv").head(1000).copy()
df["Hairpin_RNA"] = df["Hairpin_RNA"].astype(str).str.replace(" ", "", regex=False)
# Simulate dot-bracket structures (since RNAfold cannot be used here)
df["Secondary_Structure"] = ["." * len(seq) for seq in df["Hairpin_RNA"]]

# Create binary label
df["label"] = (df["Mutations"] >= 5).astype(int)

# Encoding scheme
seq_vocab = {'A': 0, 'U': 1, 'G': 2, 'C': 3, '.': 4, '(': 5, ')': 6}
def encode_seq(seq):
    return [seq_vocab.get(char, 7) for char in seq]

max_len = max(df["Hairpin_RNA"].apply(len))
df["encoded_seq"] = df["Hairpin_RNA"].apply(lambda x: encode_seq(x) + [0]*(max_len - len(x)))
df["encoded_struct"] = df["Secondary_Structure"].apply(lambda x: encode_seq(x) + [0]*(max_len - len(x)))

# Prepare tensors
X_seq = torch.tensor(df["encoded_seq"].tolist(), dtype=torch.long)
X_struct = torch.tensor(df["encoded_struct"].tolist(), dtype=torch.long)
y_binary = torch.tensor(df["label"].tolist(), dtype=torch.long)
y_reg = torch.tensor(df["Mutations"].tolist(), dtype=torch.float32)

# Split
X_seq_train, X_seq_test, X_struct_train, X_struct_test, y_bin_train, y_bin_test, y_reg_train, y_reg_test = train_test_split(
    X_seq, X_struct, y_binary, y_reg, test_size=0.2, random_state=42
)

# Dataset
class HairpinDataset(Dataset):
    def __init__(self, seqs, structs, labels):
        self.seqs = seqs
        self.structs = structs
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.seqs[idx], self.structs[idx], self.labels[idx]

train_ds = HairpinDataset(X_seq_train, X_struct_train, y_bin_train)
test_ds = HairpinDataset(X_seq_test, X_struct_test, y_bin_test)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# Model
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size=8, embed_dim=16, num_classes=2):
        super(CNNClassifier, self).__init__()
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

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses = []
val_accuracies = []

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

    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for seqs, structs, labels in test_loader:
            seqs, structs = seqs.to(device), structs.to(device)
            outputs = model(seqs, structs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    val_accuracies.append(acc)

# Output results
results_df = pd.DataFrame({
    "Epoch": list(range(1, 6)),
    "Train_Loss": train_losses,
    "Validation_Accuracy": val_accuracies
})

print(results_df)

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


