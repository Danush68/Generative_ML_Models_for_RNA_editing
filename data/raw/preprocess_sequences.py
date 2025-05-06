
import pandas as pd
import numpy as np
import os

def one_hot_encode(seq, max_len):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    encoded = np.zeros((max_len, 4))
    for i, base in enumerate(seq[:max_len]):
        if base in mapping:
            encoded[i, mapping[base]] = 1
    return encoded.flatten()

def preprocess_csv(input_csv, output_path, sequence_column="Hairpin_RNA"):
    df = pd.read_csv(input_csv)
    sequences = df[sequence_column].astype(str).str.upper().tolist()
    max_len = max(len(seq) for seq in sequences)
    encoded = np.array([one_hot_encode(seq, max_len) for seq in sequences])
    np.save(output_path, encoded)
    print(f"Saved encoded sequences to: {output_path}")

if __name__ == "__main__":
    input_csv = "../../data/raw/hairpin_rna_random_mutations.csv"  # adjust path if needed
    output_npy = "../../data/processed_sequences.npy"
    preprocess_csv(input_csv, output_npy, sequence_column="Hairpin_RNA")
