def load_fasta(filepath):
    with open(filepath, "r") as f:
        lines = f.readlines()
    return ''.join(line.strip() for line in lines if not line.startswith('>'))

def find_nth_atg(sequence, n):
    idx = -1
    count = 0
    while count < n:
        idx = sequence.find("ATG", idx + 1)
        if idx == -1:
            return None
        count += 1
    return idx

def complement_to_guide_rna(sequence):
    mapping = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(mapping.get(n, 'N') for n in sequence)

# Load full sequence
fasta_file = "LRRK2 G2019S.txt"
seq = load_fasta(fasta_file)

# Get 2019th ATG position
atg_index = find_nth_atg(seq, 2019)
if atg_index is None or atg_index < 8 or atg_index + 3 + 21 > len(seq):
    raise ValueError("ATG not found or extraction out of bounds.")

# Extract 8 upstream + ATG + 21 downstream (total = 32 bases)
target_region = seq[atg_index - 8 : atg_index + 3 + 21]
guide_rna = complement_to_guide_rna(target_region)

print("🧬 Coding Region (8 + ATG + 21):", target_region)
print("🎯 Guide RNA (complement):", guide_rna)
