import pandas as pd
import matplotlib.pyplot as plt

# === Load dataset
df = pd.read_csv("vienna_rna_full_features.csv")

# === Normalize ΔG if not already done
dg_min = df["Delta_G_MFE"].min()
dg_max = df["Delta_G_MFE"].max()
df["Normalized_DG"] = (df["Delta_G_MFE"] - dg_min) / (dg_max - dg_min)

# === Create bins if not already present
if "ΔG_Bin" not in df.columns:
    df["ΔG_Bin"] = pd.cut(df["Normalized_DG"], bins=10)

# === Summary table
summary = df.groupby("ΔG_Bin").agg(
    Raw_Count=("Sample_Weight", "count"),
    Total_Weight=("Sample_Weight", "sum"),
    Mean_Weight=("Sample_Weight", "mean")
).reset_index()

# === Save summary table
summary.to_csv("dg_bin_summary.csv", index=False)

# === Plot 1: Raw sample count per bin
plt.figure(figsize=(10, 5))
df["ΔG_Bin"].value_counts().sort_index().plot(kind="bar", color="skyblue", edgecolor="black")
plt.title("ΔG Bin Distribution (Raw Sample Count)")
plt.xlabel("ΔG Bin (Normalized)")
plt.ylabel("Number of Samples")
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("dg_bin_raw_sample_count.png")
plt.close()

# === Plot 2: Weighted contribution per bin
plt.figure(figsize=(10, 5))
summary.set_index("ΔG_Bin")["Total_Weight"].plot(kind="bar", color="green", edgecolor="black")
plt.title("Effective Sample Contribution (Total Weight per ΔG Bin)")
plt.xlabel("ΔG Bin (Normalized)")
plt.ylabel("Total Sample Weight")
plt.xticks(rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("dg_bin_weighted_contribution.png")
plt.close()

# === Plot 3: Sample weight vs ΔG (optional)
if "Sample_Weight" in df.columns:
    plt.figure(figsize=(8, 5))
    plt.scatter(df["Normalized_DG"], df["Sample_Weight"], alpha=0.3, s=10)
    plt.title("Sample Weight vs Normalized ΔG")
    plt.xlabel("Normalized ΔG")
    plt.ylabel("Sample Weight")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dg_vs_weight_scatter.png")
    plt.close()

print("✅ All plots and summary table generated.")
