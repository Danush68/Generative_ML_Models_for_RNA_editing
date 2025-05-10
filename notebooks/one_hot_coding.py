#import numpy as np
import pandas as pd
import torch
from torch.nn.functional import one_hot
from tqdm import tqdm

# Load dataset
df = pd.read_csv("../data/processed/vienna_rna_full_features.csv")

# Set target column and parameters
sequences = df["Mutated_gRNA"].str.replace("T", "U")  # ensure RNA format
max_len_ = 30  # fixed sequence length
vocab = ['A', 'C', 'G', 'U']
base_to_int = {b: i for i, b in enumerate(vocab)}

def encode_sequence(seq, max_len):
    seq = seq.upper()[:max_len].ljust(max_len, 'U')  # truncate/pad
    int_seq = [base_to_int.get(base, 3) for base in seq]  # default to 'U' for unknowns
    return one_hot(torch.tensor(int_seq), num_classes=4).float()

# Apply encoding to all sequences
encoded_tensors = torch.stack([encode_sequence(seq, max_len_) for seq in tqdm(sequences)])

# Save for training
torch.save(encoded_tensors, "../data/processed/mutated_gRNA_onehot.pt")

print(f"✅ Saved {encoded_tensors.shape[0]} sequences of shape {encoded_tensors.shape[1:]} to mutated_gRNA_onehot.pt")
