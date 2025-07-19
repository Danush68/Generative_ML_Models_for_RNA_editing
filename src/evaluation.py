import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === File Paths
input_csv = "../outputs/all_samples_per_dg.csv"
best_csv = "../outputs/best_sample_per_dg.csv"
plot_dir = "../outputs/plots/"
os.makedirs(plot_dir, exist_ok=True)

# === Load Data
df = pd.read_csv(input_csv)

# === 1. Boxplot of MFE per ΔG bin
plt.figure(figsize=(10, 6))
sns.boxplot(x="Conditioned_ΔG_Norm", y="Computed_MFE", data=df, palette="viridis")
plt.title("MFE Distribution per Conditioned ΔG Bin")
plt.xlabel("Conditioned ΔG (normalized)")
plt.ylabel("Computed MFE (kcal/mol)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/boxplot_mfe_per_dg_bin.png")
plt.close()

# === 2. Best Sample per Bin
best_per_bin = df.loc[df.groupby("Conditioned_ΔG_Norm")["Abs_Error_DG_vs_MFE"].idxmin()]
best_per_bin.to_csv(best_csv, index=False)

# Plot Best MFE vs Conditioned ΔG
plt.figure(figsize=(10, 5))
plt.plot(best_per_bin["Conditioned_ΔG_Norm"], best_per_bin["Real_ΔG_Target"], label="Target ΔG", marker='o')
plt.plot(best_per_bin["Conditioned_ΔG_Norm"], best_per_bin["Computed_MFE"], label="Best MFE", marker='x')
plt.xlabel("Conditioned ΔG (normalized)")
plt.ylabel("ΔG / MFE (kcal/mol)")
plt.title("Best Sequence per ΔG Bin: Conditioned vs Computed MFE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/best_vs_target_mfe.png")
plt.close()

# === 3. Mutation Count Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(x="Conditioned_ΔG_Norm", y="Mutation_Count", data=df, palette="coolwarm")
plt.title("Mutation Count per ΔG Bin")
plt.xlabel("Conditioned ΔG (normalized)")
plt.ylabel("Mutation Count")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/mutation_count_per_dg_bin.png")
plt.close()

# === 4. Absolute Error Distribution
plt.figure(figsize=(10, 5))
sns.violinplot(x="Conditioned_ΔG_Norm", y="Abs_Error_DG_vs_MFE", data=df, palette="pastel")
plt.title("Absolute Error Distribution per ΔG Bin")
plt.xlabel("Conditioned ΔG (normalized)")
plt.ylabel("|ΔG - MFE|")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{plot_dir}/error_distribution_per_dg_bin.png")
plt.close()

# === 5. Correlation Summary
corr_mfe = df["Conditioned_ΔG_Norm"].corr(df["Computed_MFE"])
corr_error = df["Conditioned_ΔG_Norm"].corr(df["Abs_Error_DG_vs_MFE"])

print(f"🔍 Correlation (Conditioned ΔG vs Computed MFE): {corr_mfe:.3f}")
print(f"🔍 Correlation (Conditioned ΔG vs Absolute Error): {corr_error:.3f}")
print(f"✅ Saved best samples to: {best_csv}")
print(f"✅ All plots saved in: {plot_dir}")
