import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Define the CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=5)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.fc = nn.Linear(16 * 3, 2)  # Adjusted based on sequence length

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


# Initialize model, optimizer, and loss function
model = SimpleCNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Store loss values
num_epochs = 10
loss_values = []

# Training Loop
for epoch in range(num_epochs):
    optimizer.zero_grad()  # Reset gradients
    input_data = torch.randn(10, 4, 10)  # Random batch (10 samples, 4 channels, 10-length sequence)
    output = model(input_data)  # Forward pass
    target = torch.randint(0, 2, (10,))  # Random labels (binary classification)

    loss = loss_fn(output, target)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    loss_values.append(loss.item())  # Store loss
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Plot Loss vs. Epoch
plt.plot(range(1, num_epochs + 1), loss_values, marker='o', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.grid()
plt.show()
