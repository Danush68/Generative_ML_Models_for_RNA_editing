import numpy as np

def one_hot_encode_rna(seq):
    """
    One-hot encodes a 30-nt RNA sequence into a (30, 4) matrix.
    Rows = positions (30), Columns = channels [A, C, G, U]
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

    rows = []
    for base in seq:
        if base not in mapping:
            raise ValueError(f"Invalid nucleotide '{base}' in sequence.")
        rows.append(mapping[base])

    # Return shape: (30, 4) instead of flat (120,)
    return np.array(rows, dtype=np.float32)

# Example usage:
if __name__ == "__main__":
    sequence = "GACTGATGACGTAACGAGTCATGACGACAT"  # Example 30-nt antisense RNA
    encoded_matrix = one_hot_encode_rna(sequence)

    print("Encoded Target RNA matrix:\n", encoded_matrix)
    print("Shape:", encoded_matrix.shape)  # -> (30, 4)
