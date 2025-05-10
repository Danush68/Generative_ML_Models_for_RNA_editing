import pandas as pd
import RNA
from tqdm import tqdm

# Load dataset
df = pd.read_csv("../../data/raw/hairpin_rna_random_mutations.csv")

# Add row ID to preserve order
df['__row_id'] = range(len(df))

# Convert DNA to RNA
df['Hairpin_RNA_corrected'] = df['Hairpin_RNA'].str.replace('T', 'U')

# Containers for ViennaRNA features
mfe_structures = []
mfe_energies = []
ensemble_energies = []
centroid_structures = []
centroid_energies = []
ensemble_diversities = []
freq_mfe_structs = []

# Extract ViennaRNA features
for seq in tqdm(df['Hairpin_RNA_corrected'], desc="Extracting ViennaRNA features"):
    fc = RNA.fold_compound(seq)
    mfe_struct, mfe_energy = fc.mfe()
    ensemble_energy = fc.pf()
    mfe_freq = fc.pr_structure(mfe_struct)
    diversity = fc.mean_bp_distance()
    centroid_struct, centroid_energy = fc.centroid()

    mfe_structures.append(mfe_struct)
    mfe_energies.append(mfe_energy)
    ensemble_energies.append(ensemble_energy)
    freq_mfe_structs.append(mfe_freq)
    ensemble_diversities.append(diversity)
    centroid_structures.append(centroid_struct)
    centroid_energies.append(centroid_energy)

# Add features back to the DataFrame
df['MFE Structure'] = mfe_structures
df['Delta_G_MFE'] = mfe_energies
df['Ensemble Energy'] = ensemble_energies
df['MFE Frequency'] = freq_mfe_structs
df['Ensemble Diversity'] = ensemble_diversities
df['Centroid Structure'] = centroid_structures
df['Centroid Energy'] = centroid_energies

# Sort by original row order and drop helper column
df = df.sort_values('__row_id').drop(columns='__row_id')

# Save final file
df.to_csv("../../data/processed/vienna_rna_full_features.csv", index=False)
print("✅ Saved all thermodynamic features in correct order to 'vienna_rna_full_features.csv'")