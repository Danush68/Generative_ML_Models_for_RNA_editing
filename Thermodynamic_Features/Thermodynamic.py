import pandas as pd
import RNA
from tqdm import tqdm

# Load your dataset
df = pd.read_csv("hairpin_rna_random_mutations.csv")

# Convert Mutated_Complement from DNA to RNA (T -> U)
df['Mutated_RNA'] = df['Mutated_Complement'].str.replace('T', 'U')

# Prepare containers for features
mfe_structures = []
mfe_energies = []
ensemble_energies = []
centroid_structures = []
centroid_energies = []
ensemble_diversities = []
freq_mfe_structs = []

# Loop over each RNA sequence
for seq in tqdm(df['Mutated_RNA'], desc="Extracting ViennaRNA features"):
    fc = RNA.fold_compound(seq)

    # MFE structure and energy
    mfe_struct, mfe_energy = fc.mfe()
    mfe_structures.append(mfe_struct)
    mfe_energies.append(mfe_energy)

    # Partition function and ensemble energy
    ensemble_energy = fc.pf()
    ensemble_energies.append(ensemble_energy)

    # Frequency of MFE structure in ensemble
    mfe_freq = fc.pr_structure(mfe_struct)
    freq_mfe_structs.append(mfe_freq)

    # Ensemble diversity
    diversity = fc.mean_bp_distance()
    ensemble_diversities.append(diversity)

    # Centroid structure and its energy
    centroid_struct, centroid_energy = fc.centroid()
    centroid_structures.append(centroid_struct)
    centroid_energies.append(centroid_energy)

# Add all features to the DataFrame
df['MFE Structure'] = mfe_structures
df['Delta_G_MFE'] = mfe_energies
df['Ensemble Energy'] = ensemble_energies
df['MFE Frequency'] = freq_mfe_structs
df['Ensemble Diversity'] = ensemble_diversities
df['Centroid Structure'] = centroid_structures
df['Centroid Energy'] = centroid_energies

# Save the enriched DataFrame to a CSV
df.to_csv("vienna_rna_full_features.csv", index=False)
print("✅ Saved all thermodynamic features to 'vienna_rna_full_features.csv'")
