import numpy as np

def one_hot_encode_rna(seq):
    """
    One-hot encodes a 30-nt RNA sequence into a 120-bit binary vector.
    A = [1, 0, 0, 0]
    C = [0, 1, 0, 0]
    G = [0, 0, 1, 0]
    U = [0, 0, 0, 1]
    """
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'U': [0, 0, 0, 1]
    }

    # Convert Ts to Us if needed
    seq = seq.upper().replace("T", "U")

    if len(seq) != 30:
        raise ValueError("Input sequence must be 30 nucleotides long.")

    encoded = []
    for base in seq:
        if base not in mapping:
            raise ValueError(f"Invalid nucleotide '{base}' in sequence.")
        encoded.extend(mapping[base])

    return np.array(encoded)

# Example usage:
sequence = "GACTGATGACGTAACGAGTCATGACGACAT"  # Example 30-nt antisense RNA
encoded_vector = one_hot_encode_rna(sequence)

print("Encoded Target RNA vector:\n", encoded_vector)
print("Length:", len(encoded_vector))
