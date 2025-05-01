import random

def generate_rna_sequence(length, target_codon="AUG", target_position=8):
    bases = ["A", "U", "G", "C"]
    sequence = [random.choice(bases) for _ in range(length)]

    if target_position + len(target_codon) <= length:
        sequence[target_position:target_position + 3] = list(target_codon)

    return "".join(sequence)

def mutate_rna_sequence(seq, n):
    nucleotides = ['A', 'T', 'C', 'G']
    seq_list = list(seq)
    # Create a list of indices excluding 7
    valid_positions = [i for i in range(len(seq)) if i != 8]
    positions = random.sample(valid_positions, n)

    for pos in positions:
        original = seq_list[pos]
        new_nucleotide = random.choice([nt for nt in nucleotides if nt != original])
        seq_list[pos] = new_nucleotide

    return ''.join(seq_list)

def string_rev(seq):
    return seq[::-1]

def complement_rna(sequence):
    complement_map = {"A": "T", "T": "A", "G": "C", "C": "G"}
    return "".join(complement_map[base] for base in sequence)

def create_hairpin(original, complimentary):
    loop_sequence = " TTTTT "
    compli = string_rev(complimentary)
    return original + loop_sequence + compli

def example_usage():
    # Example usage
    rna_length = 8 + 3 + 21  # 8 upstream, AUG, 21 downstream
    target_codon = "ATG"
    target_position = 8  # Position where AUG should be preserved

    #original_rna = generate_rna_sequence(rna_length, target_codon, target_position)
    #This sequence has to be fixed once the model target is obtained
    original_rna = "CTGACTACAGCATTGCTCAGTACTGCTGTA"
    complementary_rna = complement_rna(original_rna)

    complementary_rna_1 = list(complementary_rna)
    complementary_rna_1[8] = 'A' # for the mutation
    complementary_rna_1 = ''.join(complementary_rna_1)
    complementary_rna = complementary_rna_1

    mutated_rna_1 = mutate_rna_sequence(complementary_rna,1)
    mutated_rna_3 = mutate_rna_sequence(complementary_rna,3)
    hairpin_rna = create_hairpin(original_rna, complementary_rna)
    hairpin_rna_mutated_1 = create_hairpin(original_rna, mutated_rna_1)
    hairpin_rna_mutated_3 = create_hairpin(original_rna, mutated_rna_3)

    print("Original RNA:", original_rna)
    print("Complimentary RNA:", complementary_rna)
    print("Hairpin RNA:", hairpin_rna)
    print("Hairpin RNA with 1 mutated:", hairpin_rna_mutated_1)
    print("Hairpin RNA with 3 mutated:", hairpin_rna_mutated_3)

x = input("Enter 'y' if you want to see an example usage: ")
if x == 'y':
    example_usage()