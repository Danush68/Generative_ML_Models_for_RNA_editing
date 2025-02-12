import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Define the CNN Model for RNA Editing Prediction
class ADARCNN(nn.Module):
    def __init__(self, input_size=4, seq_length=50, num_filters=64, kernel_size=5, hidden_dim=128):
        super(ADARCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=2)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size,
                               padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)

        # Fully connected layers
        self.fc1 = nn.Linear((seq_length // 2) * (num_filters * 2), hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # Outputs: [Editing Efficiency, Specificity Score]

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# Custom Dataset for RNA Sequences
class RNADataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences  # One-hot encoded sequences
        self.labels = labels  # Corresponding labels (Editing Efficiency, Specificity Score)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx],
                                                                                    dtype=torch.float32)


# Function to One-Hot Encode RNA Sequences
def one_hot_encode(sequence, seq_length=50):
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1]}
    encoded_seq = np.zeros((4, seq_length))
    for i, nucleotide in enumerate(sequence[:seq_length]):
        encoded_seq[:, i] = mapping.get(nucleotide, [0, 0, 0, 0])  # Default to zeros for unknown characters
    return encoded_seq


# Example usage
model = ADARCNN()
print(model)
