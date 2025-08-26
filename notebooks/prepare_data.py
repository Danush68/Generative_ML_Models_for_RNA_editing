
# prepare_data.py (robust "Mutations" fetch)
#
# Reads your GRNADataset (from vienna_rna_full_features.csv), splits 80/10/10,
# and writes .pt files that include mut_count (int) and mut_bin (one-hot, 5 bins).
# If the Subset item dict doesn't include "Mutations", we look it up from the
# underlying dataset's dataframe by index.
#
# Works with torch.utils.data.random_split which returns Subset objects.

from grna_dataset import GRNADataset
from torch.utils.data import random_split, Subset
import torch
import os

# === Config (must match train/sample) ===
MUT_BIN_EDGES = [0, 2, 4, 6, 8, 10]   # (0-2], (2-4], (4-6], (6-8], (8-10]
MUT_DIM = len(MUT_BIN_EDGES) - 1

def bucketize_count_tensor(count: torch.Tensor) -> torch.Tensor:
    bins = torch.bucketize(count.float(),
                           torch.tensor(MUT_BIN_EDGES, dtype=torch.float32),
                           right=True) - 1
    bins = bins.clamp(0, MUT_DIM - 1).to(torch.long)
    return torch.nn.functional.one_hot(bins, num_classes=MUT_DIM).float()

os.makedirs("../data/processed", exist_ok=True)

# Load dataset
base = GRNADataset("../data/processed/vienna_rna_full_features.csv")

# 80/10/10 split
N = len(base)
n_train = int(0.8 * N)
n_val = int(0.1 * N)
n_test = N - n_train - n_val

train_set, val_set, test_set = random_split(base, [n_train, n_val, n_test])

def get_mutations_from_base(ds, idx):
    # Try common attribute names for the backing dataframe
    for attr in ["df", "dataframe", "data", "table"]:
        if hasattr(ds, attr):
            df = getattr(ds, attr)
            try:
                return int(df.iloc[idx]["Mutations"])
            except Exception:
                pass
    # If dataset exposes original CSV rows as list of dicts
    if hasattr(ds, "__getitem__"):
        row = ds[idx]
        # try alternative keys
        for key in ["Mutations", "mutations", "mut_count", "mutation_count"]:
            if isinstance(row, dict) and key in row:
                return int(row[key])
    raise KeyError(f"Could not retrieve 'Mutations' for index {idx}; ensure your GRNADataset exposes it or underlying dataframe has the column.")

def save_split(split: Subset, name: str):
    X, CSEQ, CDG, W = [], [], [], []
    MC = []

    # Access the underlying dataset and indices to recover row ids
    underlying = split.dataset
    indices = split.indices

    warn_missing_once = True

    for i, idx in enumerate(indices):
        item = underlying[idx]  # pull tensors from dataset

        X.append(item["x"])
        CSEQ.append(item["cond_seq"])
        CDG.append(item["cond_dg"])
        W.append(item.get("sample_weight", torch.tensor(1.0)))

        # Prefer direct key if provided
        if isinstance(item, dict) and "Mutations" in item:
            MC.append(torch.tensor(int(item["Mutations"]), dtype=torch.int32))
        elif isinstance(item, dict) and "mut_count" in item:
            MC.append(torch.tensor(int(item["mut_count"]), dtype=torch.int32))
        else:
            # Fallback: pull from underlying dataframe by original index
            if warn_missing_once:
                print("⚠️  'Mutations' not in item — pulling from base dataset by index...")
                warn_missing_once = False
            mval = get_mutations_from_base(underlying, idx)
            MC.append(torch.tensor(int(mval), dtype=torch.int32))

    x = torch.stack(X)
    cond_seq = torch.stack(CSEQ)
    cond_dg = torch.stack(CDG)
    sample_weight = torch.stack(W)
    mut_count = torch.stack(MC).view(-1).to(torch.float32)
    mut_bin = bucketize_count_tensor(mut_count)

    torch.save({
        "x": x,
        "cond_seq": cond_seq,
        "cond_dg": cond_dg,
        "sample_weight": sample_weight,
        "mut_count": mut_count.to(torch.int32),
        "mut_bin": mut_bin,
    }, f"../data/processed/{name}.pt")
    print(f"✅ Saved {name}.pt — {len(x)} samples (with Mutations → mut_count & mut_bin)")

save_split(train_set, "train")
save_split(val_set, "val")
save_split(test_set, "test")
