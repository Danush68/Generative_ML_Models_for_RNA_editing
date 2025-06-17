# prepare_data.py

from grna_dataset import GRNADataset
from torch.utils.data import random_split
import torch
import os

# Make sure the output folder exists
os.makedirs("../data/processed", exist_ok=True)

# Load dataset
dataset = GRNADataset("../data/processed/vienna_rna_full_features.csv")

# Split dataset (80% train, 10% val, 10% test)
total = len(dataset)
train_size = int(0.8 * total)
val_size = int(0.1 * total)
test_size = total - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# Save split datasets as .pt
def save_dataset(split, name):
    x_list, cond_seq_list, cond_dg_list = [], [], []
    for item in split:
        x_list.append(item["x"])
        cond_seq_list.append(item["cond_seq"])
        cond_dg_list.append(item["cond_dg"])
    torch.save({
        "x": torch.stack(x_list),
        "cond_seq": torch.stack(cond_seq_list),
        "cond_dg": torch.stack(cond_dg_list),
    }, f"../data/processed/{name}.pt")
    print(f"✅ Saved {name}.pt — {len(x_list)} samples")

# Run save for all splits
save_dataset(train_set, "train")
save_dataset(val_set, "val")
save_dataset(test_set, "test")
