import pandas as pd
import torch
from torch.nn.functional import one_hot

# Load CSV
df = pd.read_csv("../data/processed/vienna_rna_full_features.csv")
sequences = df["Original_RNA"].str.replace("T", "U")

# One-hot encoding
vocab = ['A', 'C', 'G', 'U']
base_to_int = {base: i for i, base in enumerate(vocab)}
max_len = 30

def encode(seq):
    seq = seq.upper()[:max_len].ljust(max_len, 'U')
    int_seq = [base_to_int.get(b, 3) for b in seq]
    return one_hot(torch.tensor(int_seq), num_classes=4).float()

tensor = torch.stack([encode(seq) for seq in sequences])
torch.save(tensor, "../data/processed/target_onehot.pt")
print("✅ Saved target_onehot.pt with shape", tensor.shape)
