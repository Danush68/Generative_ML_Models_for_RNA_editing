# model.py
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, vocab_size=4, embed_dim=8, seq_length=50, num_channels=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # A=0, U=1, C=2, G=3
        self.conv_layers = nn.Sequential(
            nn.Conv1d(embed_dim, num_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        self.fc = nn.Sequential(
            nn.Linear(num_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # Output: efficiency + specificity
        )

    def forward(self, gRNA_seq, target_seq):
        # Input shape: [batch_size, seq_length]
        embedded = self.embedding(torch.cat([gRNA_seq, target_seq], dim=1))  # [batch, 2*seq_length, embed_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch, embed_dim, 2*seq_length]
        features = self.conv_layers(embedded).squeeze(-1)
        return self.fc(features)