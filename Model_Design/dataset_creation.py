import random
import pandas as pd

def string_rev(seq):
    return seq[::-1]

def complement_rna(sequence):
    complement_map = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join(complement_map[base] for base in sequence)

def mutate_rna_sequence(seq, n):
    nucleotides = ['A', 'T', 'C', 'G']
    seq_list = list(seq)
    # Create a list of indices excluding 7
    valid_positions = [i for i in range(len(seq)) if i != 7]
    positions = random.sample(valid_positions, n)

    for pos in positions:
        original = seq_list[pos]
        new_nucleotide = random.choice([nt for nt in nucleotides if nt != original])
        seq_list[pos] = new_nucleotide

    return ''.join(seq_list)

def create_hairpin(original, complementary):
    loop_sequence = " TTTTT "
    compli = string_rev(complementary)
    return original + loop_sequence + compli

# 🔬 Fixed original RNA sequence (centered around 2019th mutation)
original_rna = "CTGACTACAGCATTGCTCAGTACTGCTGTA"
complementary_rna = complement_rna(original_rna)

# 📦 Generate 10,000 mutated samples with random 1–15 mutations
samples = []
num_samples = 10000

complementary_rna_1 = list(complementary_rna)
complementary_rna_1[8] = 'A' # for the mutation
complementary_rna_1 = ''.join(complementary_rna_1)
complementary_rna = complementary_rna_1

for _ in range(num_samples):
    num_mutations = random.randint(1, 15)
    mutated_compl = mutate_rna_sequence(complementary_rna, num_mutations)
    mutated_hairpin = create_hairpin(original_rna, mutated_compl)
    samples.append({
        "Original_RNA": original_rna,
        "Original_Compliment(GuideRNA)": complementary_rna,
        "Mutated_Complement": mutated_compl,
        "Mutations": num_mutations,
        "Hairpin_RNA": mutated_hairpin,
    })

# 💾 Save to CSV
df = pd.DataFrame(samples)
df.to_csv("hairpin_rna_random_mutations.csv", index=False)
print("✅ Saved 10,000 hairpin RNAs with 1–15 mutations to 'hairpin_rna_random_mutations.csv'")
