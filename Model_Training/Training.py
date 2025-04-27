import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("../Model_Design/hairpin_rna_random_mutations.csv")

# Create binary labels: 1 if mutations >= 5
df["label"] = (df["Mutations"] >= 5).astype(int)
df["clean_seq"] = df["Hairpin_RNA"].apply(lambda x: x.replace(" ", ""))

# Encode sequences
vocab = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'T': 4, 'N': 5}
df["encoded_seq"] = df["clean_seq"].apply(lambda seq: [vocab.get(base, 5) for base in seq])
MAX_LEN = max(df["encoded_seq"].apply(len))
df["padded_seq"] = df["encoded_seq"].apply(lambda x: x + [0] * (MAX_LEN - len(x)))

# Split dataset
train_df, val_df = train_test_split(df[["padded_seq", "label"]], test_size=0.2, stratify=df["label"], random_state=42)

# Dataset class
class RNASequenceDataset(Dataset):
    def __init__(self, df):
        self.sequences = df["padded_seq"].tolist()
        self.labels = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Dataloaders
train_loader = DataLoader(RNASequenceDataset(train_df), batch_size=32, shuffle=True)
val_loader = DataLoader(RNASequenceDataset(val_df), batch_size=32)

# BiLSTM model
class RNABiLSTM(nn.Module):
    def __init__(self, vocab_size=6, embed_dim=16, hidden_dim=64, output_dim=2):
        super(RNABiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        hn = torch.cat((hn[0], hn[1]), dim=1)
        return self.fc(hn)

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNABiLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
writer = SummaryWriter(log_dir="runs/hairpin_rna_classification")

# Training and validation
train_losses, val_accuracies = [], []

def validate(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    return accuracy_score(all_labels, all_preds)

def train(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = validate(model, val_loader)

        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)

        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}, Val Accuracy = {val_acc:.4f}")

# Run training
train(model, train_loader, val_loader, epochs=10)
writer.close()

# Plot results
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Validation Accuracy", color="orange")
plt.title("Validation Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# Get predictions for validation set
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(device)
        outputs = model(x_batch)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.numpy())

