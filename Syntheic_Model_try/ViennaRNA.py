from ViennaRNA import RNA

def compute_structure(gRNA, target):
    structure, energy = RNA.cofold(gRNA + '&' + target)
    return structure, energy

# Example usage in loss function
def structure_aware_loss(predicted_seq, target_seq):
    structure, energy = compute_structure(predicted_seq, target_seq)
    # Penalize high free energy (unstable structures)
    return torch.abs(energy - ideal_energy)