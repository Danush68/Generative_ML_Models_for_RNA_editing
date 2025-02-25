import random


def generate_rna_sequence(length, target_codon="AUG", target_position=8):
    """
    Generates a random RNA sequence of given length while ensuring the target codon
    appears at the specified position.
    """
    bases = ["A", "U", "G", "C"]
    sequence = [random.choice(bases) for _ in range(length)]

    if target_position + len(target_codon) <= length:
        sequence[target_position:target_position + 3] = list(target_codon)

    return "".join(sequence)


def mutate_rna(sequence, exclude_start, exclude_end, mutation_rate=0.1):
    """
    Introduces random mutations in the RNA sequence except in the specified region.
    """
    bases = ["A", "U", "G", "C"]
    sequence = list(sequence)

    for i in range(len(sequence)):
        if exclude_start <= i < exclude_end:
            continue  # Skip mutation in the AUG region

        if random.random() < mutation_rate:
            sequence[i] = random.choice([b for b in bases if b != sequence[i]])

    return "".join(sequence)


def complement_rna(sequence):
    """
    Returns the complementary RNA sequence.
    """
    complement_map = {"A": "U", "U": "A", "G": "C", "C": "G"}
    return "".join(complement_map[base] for base in sequence)


# Example usage
rna_length = 8 + 3 + 21  # 8 upstream, AUG, 21 downstream
target_codon = "AUG"
target_position = 8  # Position where AUG should be preserved

original_rna = generate_rna_sequence(rna_length, target_codon, target_position)
mutated_rna = mutate_rna(original_rna, target_position, target_position + 3)
complementary_rna = complement_rna(mutated_rna)

print("Original RNA:", original_rna)
print("Mutated RNA:", mutated_rna)
print("Complementary RNA:", complementary_rna)
