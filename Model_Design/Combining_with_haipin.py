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


def complement_rna(sequence):
    """
    Returns the complementary RNA sequence.
    """
    complement_map = {"A": "U", "U": "A", "G": "C", "C": "G"}
    return "".join(complement_map[base] for base in sequence)


def create_hairpin(original, complimentary):
    """
    Creates a structured hairpin loop including barcode, variable region, and hairpin.
    """
    barcode = "GGGAAACCC"  # Example barcode sequence
    invariant_left = "GGG"  # Left invariant region
    invariant_right = "CCC"  # Right invariant region
    loop_sequence = "UUUUU"  # Hairpin loop structure

    return original + loop_sequence + complimentary


# Example usage
rna_length = 8 + 3 + 21  # 8 upstream, AUG, 21 downstream
target_codon = "AUG"
target_position = 8  # Position where AUG should be preserved

original_rna = generate_rna_sequence(rna_length, target_codon, target_position)
complementary_rna = complement_rna(original_rna)
hairpin_rna = create_hairpin(original_rna, complementary_rna)

print("Original RNA:", original_rna)
print("Complimentary RNA:", complementary_rna)
print("Hairpin RNA:", hairpin_rna)
