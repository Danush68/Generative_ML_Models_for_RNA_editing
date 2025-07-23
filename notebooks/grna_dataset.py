# dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

NUC_TO_ONE_HOT = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1]}
NUC_TO_BITS = {'A': [0, 0], 'C': [0, 1], 'G': [1, 0], 'U': [1, 1]}

class GRNADataset(Dataset):
    def __init__(self, csv_path, dg_min=None, dg_max=None):
        self.df = pd.read_csv(csv_path)
        self.df['Mutated_gRNA'] = self.df['Mutated_gRNA'].str.replace('T', 'U')
        self.df['Original_RNA'] = self.df['Original_RNA'].str.replace('T', 'U')

        # ΔG normalization
        if dg_min is None:
            dg_min = self.df['Delta_G_MFE'].min()
        if dg_max is None:
            dg_max = self.df['Delta_G_MFE'].max()
        self.dg_min, self.dg_max = np.percentile(self.df['Delta_G_MFE'], [0, 100])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = torch.tensor([bit for nt in row['Mutated_gRNA'] for bit in NUC_TO_BITS[nt]], dtype=torch.float32)
        cond_seq = torch.tensor([NUC_TO_ONE_HOT[nt] for nt in row['Original_RNA']], dtype=torch.float32)

        dg = (row['Delta_G_MFE'] - self.dg_min) / (self.dg_max - self.dg_min)
        cond_dg = torch.tensor([min(max(dg, 0.0), 1.0)], dtype=torch.float32)

        sample_weight = row.get("Sample_Weight", 1.0)
        sample_weight = torch.tensor(sample_weight, dtype=torch.float32)

        return {
            "x": x,
            "cond_seq": cond_seq,
            "cond_dg": cond_dg,
            "sample_weight": sample_weight
        }
