import torch
import torch.nn as nn


# -------------------------
# CNN Model for Regression
# -------------------------
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
