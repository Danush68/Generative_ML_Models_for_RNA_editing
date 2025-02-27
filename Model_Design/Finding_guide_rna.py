def read_rna_sequence(file_path):
    with open(file_path, 'r') as file:
        rna_seq = file.read().strip()  # Read and remove extra spaces/newlines
    return rna_seq


def find_guide_rna(rna_seq, mutation_pos):
    upstream = 8  # Nucleotides before the mutation
    downstream = 21  # Nucleotides after the mutation

    # Ensure the mutation position is valid
    if mutation_pos - upstream < 0 or mutation_pos + downstream >= len(rna_seq):
        raise ValueError("Mutation position is too close to the sequence start or end.")

    # Extract the 29-nt guide RNA sequence
    guide_rna = rna_seq[mutation_pos - upstream: mutation_pos + downstream + 1]

    return guide_rna


file_path = "LRRK2 G2019S.txt"  # Replace with your actual file path
mutation_position = 6055 # Update with the actual 0-based index of the G2019S mutation

# Read RNA sequence from file
rna_sequence = read_rna_sequence(file_path)

# Find guide RNA
guide_rna_seq = find_guide_rna(rna_sequence, mutation_position)

# Output results
print("Original RNA Sequence:", rna_sequence)
print("Original RNA Sequence:", rna_sequence[6047:6077:1])
print("mutation position:",mutation_position)
print("Guide RNA Sequence:", guide_rna_seq)
