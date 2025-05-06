import torch
import torch.nn as nn

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






