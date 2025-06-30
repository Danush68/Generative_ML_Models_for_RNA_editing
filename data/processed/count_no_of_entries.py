import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("vienna_rna_full_features.csv")

# Compute normalized ΔG using current min/max
dg_min = df["Delta_G_MFE"].min()
dg_max = df["Delta_G_MFE"].max()
df["Normalized_DG"] = (df["Delta_G_MFE"] - dg_min) / (dg_max - dg_min)

# Bin into 10 equal buckets
df["DG_Bin"] = pd.cut(df["Normalized_DG"], bins=10)

# Count samples per bin
bin_counts = df["DG_Bin"].value_counts().sort_index()

print(bin_counts)
bin_counts.plot(kind="bar", title="ΔG Bin Distribution")
plt.ylabel("Number of Samples")
plt.xlabel("ΔG Bin (Normalized)")
plt.tight_layout()
plt.show()
