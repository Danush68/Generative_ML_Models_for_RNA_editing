# dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd

NUC_TO_ONE_HOT = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'U': [0, 0, 0, 1],
}

NUC_TO_BITS = {
    'A': [0, 0],
    'C': [0, 1],
    'G': [1, 0],
    'U': [1, 1],
}

class GRNADataset(Dataset):
    def __init__(self, csv_path, dg_min=None, dg_max=None):
        self.df = pd.read_csv(csv_path)

        # Replace T with U for RNA
        self.df['Mutated_gRNA'] = self.df['Mutated_gRNA'].str.replace('T', 'U')
        self.df['Original_RNA'] = self.df['Original_RNA'].str.replace('T', 'U')

        # ΔG normalization
        if dg_min is None:
            dg_min = self.df['Delta_G_MFE'].min()
        if dg_max is None:
            dg_max = self.df['Delta_G_MFE'].max()
        # Updated: Use 1st–99th percentile for ΔG normalization
        import numpy as np
        self.dg_min, self.dg_max = np.percentile(self.df['Delta_G_MFE'], [1, 99])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Binary-encoded gRNA (flattened list of bits)
        x_seq = [bit for nt in row['Mutated_gRNA'] for bit in NUC_TO_BITS[nt]]
        x = torch.tensor(x_seq, dtype=torch.float32)

        # One-hot Original_RNA as conditioning
        cond_seq = [NUC_TO_ONE_HOT[nt] for nt in row['Original_RNA']]
        cond_seq = torch.tensor(cond_seq, dtype=torch.float32)

        # Normalized ΔG value
        dg = (row['Delta_G_MFE'] - self.dg_min) / (self.dg_max - self.dg_min)
        dg = min(max(dg, 0.0), 1.0)  # Clip to [0, 1]
        cond_dg = torch.tensor([dg], dtype=torch.float32)

        return {
            "x": x,                       # (seq_len * 2,)
            "cond_seq": cond_seq,        # (seq_len, 4)
            "cond_dg": cond_dg           # (1,)
        }
