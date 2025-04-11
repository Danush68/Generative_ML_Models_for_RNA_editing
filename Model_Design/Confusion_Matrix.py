import random
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Complement mapping
def complement_rna(sequence):
    complement_map = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join(complement_map[base] for base in sequence)

# Reverse string
def string_rev(seq):
    return seq[::-1]

# Mutate sequence with `n` random point mutations
def mutate_rna_sequence(seq, n):
    nucleotides = ['A', 'T', 'C', 'G']
    seq_list = list(seq)
    positions = random.sample(range(len(seq)), n)
    for pos in positions:
        original = seq_list[pos]
        seq_list[pos] = random.choice([nt for nt in nucleotides if nt != original])
    return ''.join(seq_list), positions

# Create hairpin by connecting sequence and its reverse-complement with loop
def create_hairpin(original, complementary):
    loop_sequence = " TTTTT "
    return original + loop_sequence + string_rev(complementary)

# One-hot encoding
def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    return [mapping.get(nuc, [0, 0, 0, 0]) for nuc in seq]

# Reference fixed RNA sequence (based on 2019th mutation site)
original_rna = "CCTTATCAATTCATTCTAGAGAAATCTGGA"
complementary_rna = complement_rna(original_rna)

# Generate samples
samples = []
original_bases = []
mutated_bases = []
num_samples = 10000

for _ in range(num_samples):
    num_mut = random.randint(1, 15)
    mutated_compl, positions = mutate_rna_sequence(complementary_rna, num_mut)
    mutated_hairpin = create_hairpin(original_rna, mutated_compl)

    # Track base changes for confusion matrix
    for pos in positions:
        original_bases.append(complementary_rna[pos])
        mutated_bases.append(mutated_compl[pos])

    samples.append({
        "Original_RNA": original_rna,
        "Mutated_Complement": mutated_compl,
        "Mutations": num_mut,
        "Hairpin_RNA": mutated_hairpin,
        "OneHotEncoded": one_hot_encode(mutated_compl)
    })

# Create main DataFrame
df = pd.DataFrame(samples)

# Save full dataset
df.to_csv("hairpin_rna_dataset.csv", index=False)
print("✅ Saved full dataset to 'hairpin_rna_dataset.csv'")

# Save filtered CSVs by mutation count
# for i in range(1, 16):
#     df_filtered = df[df["Mutations"] == i]
#     df_filtered.to_csv(f"hairpin_rna_mutations_{i}.csv", index=False)
#
# print("✅ Saved filtered datasets (1–15 mutations each)")

# Convert confusion matrix to readable format
labels = ['A', 'T', 'C', 'G']
cm = confusion_matrix(original_bases, mutated_bases, labels=labels)
conf_df = pd.DataFrame(cm, index=labels, columns=labels)
print("\n🔬 Confusion Matrix (Original vs Mutated Bases):\n")
print(conf_df)

# Visualize the matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues')
plt.title("Nucleotide Mutation Confusion Matrix")
plt.show()
