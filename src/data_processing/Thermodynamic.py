import pandas as pd
import RNA
from tqdm import tqdm
import os

# === File Paths
input_path = "../../data/raw/hairpin_rna_random_mutations.csv"
output_path = "../../data/processed/vienna_rna_full_features.csv"

# Load dataset
df = pd.read_csv(input_path)
df['__row_id'] = range(len(df))
df['Hairpin_RNA_corrected'] = df['Hairpin_RNA'].str.replace('T', 'U')

# Extract ViennaRNA features
mfe_structures = []
mfe_energies = []

for seq in tqdm(df['Hairpin_RNA_corrected'], desc="Extracting ViennaRNA features"):
    fc = RNA.fold_compound(seq)
    struct, energy = fc.mfe()
    mfe_structures.append(struct)
    mfe_energies.append(energy)

df['MFE Structure'] = mfe_structures
df['Delta_G_MFE'] = mfe_energies
df = df.sort_values('__row_id').drop(columns='__row_id')

# Normalize and bin ΔG
min_dg = df["Delta_G_MFE"].min()
max_dg = df["Delta_G_MFE"].max()
df["Conditioned_ΔG_Norm"] = (df["Delta_G_MFE"] - min_dg) / (max_dg - min_dg)
bins = [i / 10 for i in range(11)]
df["ΔG_Bin"] = pd.cut(df["Conditioned_ΔG_Norm"], bins=bins, include_lowest=True)

# Add sample weights: inverse frequency
bin_counts = df["ΔG_Bin"].value_counts(normalize=True).to_dict()
df["Sample_Weight"] = df["ΔG_Bin"].map(lambda b: 1.0 / bin_counts.get(b, 1.0))

# Save
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"✅ Saved full dataset with Sample_Weight to: {output_path}")
