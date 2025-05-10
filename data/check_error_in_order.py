import pandas as pd

# Load original and processed data
original = pd.read_csv("../data/raw/hairpin_rna_random_mutations.csv")
processed = pd.read_csv("../data/processed/vienna_rna_full_features.csv")

# Sanity check: Compare the 'Mutated_Complement' columns to ensure row order is preserved
if "Mutated_Complement" in original.columns and "Mutated_Complement" in processed.columns:
    match = (original["Mutated_Complement"] == processed["Mutated_Complement"]).all()
    if match:
        print("✅ Row order is preserved: Mutated_Complement columns match exactly.")
    else:
        print("❌ Row mismatch detected: Mutated_Complement columns differ.")
else:
    print("⚠️ 'Mutated_Complement' column not found in both files.")

