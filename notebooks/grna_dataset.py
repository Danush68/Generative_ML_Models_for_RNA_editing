# dataset.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

NUC_TO_ONE_HOT = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1]}

class GRNADataset(Dataset):
    def __init__(self, csv_path, dg_min=None, dg_max=None):
        self.df = pd.read_csv(csv_path)
        # normalize alphabet
        self.df['Mutated_gRNA'] = self.df['Mutated_gRNA'].str.upper().str.replace('T', 'U', regex=False)
        self.df['Original_RNA'] = self.df['Original_RNA'].str.upper().str.replace('T', 'U', regex=False)

        # optional: ensure all sequences are 30 nt
        if not (self.df['Mutated_gRNA'].str.len() == 30).all():
            bad = self.df[self.df['Mutated_gRNA'].str.len() != 30].index.tolist()[:5]
            raise ValueError(f"Mutated_gRNA must be 30 nt for all rows. Offenders at indices: {bad}")
        if not (self.df['Original_RNA'].str.len() == 30).all():
            bad = self.df[self.df['Original_RNA'].str.len() != 30].index.tolist()[:5]
            raise ValueError(f"Original_RNA must be 30 nt for all rows. Offenders at indices: {bad}")

        # ΔG normalization bounds (use provided values if given; else min/max of data)
        if dg_min is None:
            dg_min = self.df['Delta_G_MFE'].min()
        if dg_max is None:
            dg_max = self.df['Delta_G_MFE'].max()
        self.dg_min, self.dg_max = float(dg_min), float(dg_max)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # x and cond_seq as (30, 4) one-hot
        x = torch.tensor([NUC_TO_ONE_HOT[nt] for nt in row['Mutated_gRNA']], dtype=torch.float32)          # (30, 4)
        cond_seq = torch.tensor([NUC_TO_ONE_HOT[nt] for nt in row['Original_RNA']], dtype=torch.float32)   # (30, 4)

        # ΔG normalized to [0,1]
        dg = (row['Delta_G_MFE'] - self.dg_min) / (self.dg_max - self.dg_min + 1e-12)
        cond_dg = torch.tensor([min(max(float(dg), 0.0), 1.0)], dtype=torch.float32)                        # (1,)

        # sample weight (defaults to 1.0 if column missing)
        sample_weight = torch.tensor(row.get("Sample_Weight", 1.0), dtype=torch.float32)

        return {
            "x": x,                     # (30, 4)
            "cond_seq": cond_seq,       # (30, 4)
            "cond_dg": cond_dg,         # (1,)
            "sample_weight": sample_weight
        }
