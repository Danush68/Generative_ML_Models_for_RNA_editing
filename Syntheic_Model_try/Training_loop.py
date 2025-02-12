import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from CNN_Ensemble import CNN  # Import the CNN class

# Sample data preparation (replace with real data)
num_samples = 1000
seq_length = 50
vocab = {'A': 0, 'U': 1, 'C': 2, 'G': 3}

# Random sequences (example only)
gRNA_seqs = torch.randint(0, 4, (num_samples, seq_length))
target_seqs = torch.randint(0, 4, (num_samples, seq_length))
efficiencies = torch.rand(num_samples)
specificities = torch.rand(num_samples)

# Create Dataset and DataLoader
dataset = TensorDataset(gRNA_seqs, target_seqs, efficiencies, specificities)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
model = CNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Track loss for plotting
epoch_losses = []

# Training loop
for epoch in range(100):
    total_loss = 0
    for batch in dataloader:
        gRNA, target, eff, spec = batch
        preds = model(gRNA, target)
        loss = criterion(preds[:, 0], eff) + criterion(preds[:, 1], spec)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    epoch_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
epoch = 100
plt.figure(figsize=(10, 6))
plt.plot(range(1, epoch + 1), epoch_losses, marker='o', linestyle='-', color='b', label='Training Loss')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Across Epochs', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
