import pandas as pd
import matplotlib.pyplot as plt

# Load full dataset
df = pd.read_csv("vienna_rna_full_features.csv")

# Normalize ΔG again (if not already done)
dg_min = df["Delta_G_MFE"].min()
dg_max = df["Delta_G_MFE"].max()
df["Normalized_DG"] = (df["Delta_G_MFE"] - dg_min) / (dg_max - dg_min)

# Bin into 10 equal-width buckets
df["DG_Bin"] = pd.cut(df["Normalized_DG"], bins=10)

# 1. Count samples per bin
bin_counts = df["DG_Bin"].value_counts().sort_index()
print("\n🧮 Sample Count per ΔG Bin:\n", bin_counts)

# 2. Plot: Sample Count per Bin
plt.figure(figsize=(10, 5))
bin_counts.plot(kind="bar", title="ΔG Bin Distribution", color="skyblue", edgecolor="black")
plt.ylabel("Number of Samples")
plt.xlabel("ΔG Bin (Normalized)")
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("dg_bin_distribution.png")
plt.show()

# 3. Plot: Sample Weight vs ΔG (Optional)
if "Sample_Weight" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Normalized_DG"], df["Sample_Weight"], alpha=0.3, s=10)
    plt.title("Sample Weight vs Normalized ΔG")
    plt.xlabel("Normalized ΔG")
    plt.ylabel("Sample Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dg_vs_weight.png")
    plt.show()
