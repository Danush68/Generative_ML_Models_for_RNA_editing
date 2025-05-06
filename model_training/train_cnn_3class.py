import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
from src.models.CNN_Model_3_Class import CNNClassifier

# -----------------------------
# Load and preprocess dataset
# -----------------------------
df = pd.read_csv("../data/raw/hairpin_rna_random_mutations.csv")
df["Hairpin_RNA"] = df["Hairpin_RNA"].astype(str).str.replace(" ", "")

# Binning mutations into 3 classes
def bin_mutation(m):
    if m <= 5:
        return 0
    elif m <= 10:
        return 1
    else:
        return 2

df["label"] = df["Mutations"].apply(bin_mutation)

# Encode RNA sequence
vocab = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
def encode(seq):
    return [vocab.get(base, 0) for base in seq]

max_len = df["Hairpin_RNA"].str.len().max()
df["encoded_seq"] = df["Hairpin_RNA"].apply(lambda x: encode(x) + [0]*(max_len - len(x)))

X = torch.tensor(df["encoded_seq"].tolist(), dtype=torch.long)
y = torch.tensor(df["label"].tolist(), dtype=torch.long)

# -----------------------------
# Dataset and Dataloader
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
# Training Loop
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_losses, val_accuracies = [], []

for epoch in range(20):
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
# Save Model
# -----------------------------
os.makedirs("../outputs/models", exist_ok=True)
torch.save(model.state_dict(), "../outputs/models/cnn_3class.pth")

# -----------------------------
# 6. Evaluation and Plots
# -----------------------------
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
plt.savefig("../outputs/plots/3class_loss_vs_accuracy.png")
plt.show()

# Classification report and confusion matrix
print("\n📊 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["0–5", "6–10", "11–15"]))

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0–5", "6–10", "11–15"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.grid(False)
plt.savefig("../outputs/confusion_matrix/3class_confusion_matrix.png")
plt.show()
