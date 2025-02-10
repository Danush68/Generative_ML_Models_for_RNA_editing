import numpy as np

def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0], 'U': [0, 1, 0, 0],
               'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    return np.array([mapping[nt] for nt in seq])

sequence = "AUGCGU"
encoded_seq = one_hot_encode(sequence)
print(encoded_seq)

