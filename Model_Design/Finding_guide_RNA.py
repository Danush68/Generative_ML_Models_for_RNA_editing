# Step 0: Load the CDS from the FASTA file
with open("FASTA Nucleotide.txt", "r") as f:
    lines = f.readlines()
    # Skip header lines starting with ">"
    cds_seq = ''.join([line.strip() for line in lines if not line.startswith(">")])

# Step 1: Apply mutation (G → A at position 6055)
cds_mut = list(cds_seq)
cds_mut[6055 - 1] = 'A'  # 0-based index
cds_mut = ''.join(cds_mut)

# Step 2: Calculate the codon position for amino acid #2019
aa_position = 2019
codon_start = (aa_position - 1) * 3  # 0-based index
target_a_index = codon_start + 1  # Middle nucleotide of codon (the editable A)

# Step 3: Extract window from -8 to +21 around the target A
upstream = 9
downstream = 18
window_start = target_a_index - upstream
window_end = target_a_index + 3 + downstream
target_window_dna = cds_mut[window_start:window_end]

# Step 4: Convert to RNA (T → U)
target_window_rna = target_window_dna.replace("T", "U")

# Step 5: Print or return the target RNA sequence
print("Target RNA sequence (-8 to +21):")
print(target_window_rna)
print(target_window_dna)
