import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


# Define the CNN Model for RNA Editing Prediction
class ADARCNN(nn.Module):
    def __init__(self, input_size=4, seq_length=50, num_filters=64, kernel_size=5, hidden_dim=128):
        super(ADARCNN, self).__init__()

        self.relu = nn.ReLU()  # Ensure ReLU is initialized before usage

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_filters, kernel_size=kernel_size, padding=2)
        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=kernel_size,
                               padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)

        # Compute dynamic input size for fc1
        self.fc1_input_dim = self._get_conv_output_size(seq_length)

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc1_input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2)  # Outputs: [Editing Efficiency, Specificity Score]

    def _get_conv_output_size(self, seq_length):
        with torch.no_grad():
            x = torch.zeros(1, 4, seq_length)  # Dummy input
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            return x.numel()

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


# Load and prepare dataset
def load_data(sequences, labels, batch_size=32):
    dataset = RNADataset(sequences, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# Training function
def train_model(model, dataloader, epochs=1000, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for sequences, labels in dataloader:
            sequences, labels = sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    print("Training complete!")

    # Plot training loss
    plt.plot(range(1, epochs + 1), losses, marker='o', linestyle='-', color='b', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Dummy data
    sample_sequences = ["ACGUACGUACGUACGUACGUACGUACGUACGUACGUACGUACGU"] * 100
    sample_labels = np.random.rand(100, 2)  # Random efficiency and specificity scores

    encoded_sequences = np.array([one_hot_encode(seq) for seq in sample_sequences])
    dataloader = load_data(encoded_sequences, sample_labels)

    model = ADARCNN()
    train_model(model, dataloader)
