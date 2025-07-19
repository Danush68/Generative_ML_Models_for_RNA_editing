import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# Load data
df = pd.read_csv("../outputs/all_samples_per_dg.csv")

# Compute error
df["abs_error"] = df["Abs_Error_DG_vs_MFE"]

# Metrics
mae = mean_absolute_error(df["Conditioned_ΔG_Norm"], df["Computed_MFE"])
mse = mean_squared_error(df["Conditioned_ΔG_Norm"], df["Computed_MFE"])
r2 = r2_score(df["Conditioned_ΔG_Norm"], df["Computed_MFE"])
corr, _ = pearsonr(df["Conditioned_ΔG_Norm"], df["Computed_MFE"])

# % within thresholds
within_2 = (df["abs_error"] <= 2).mean() * 100
within_5 = (df["abs_error"] <= 5).mean() * 100
within_10 = (df["abs_error"] <= 10).mean() * 100

# Print summary
print(f"📊 ΔG Evaluation Metrics:")
print(f" - MAE: {mae:.2f}")
print(f" - MSE: {mse:.2f}")
print(f" - R²: {r2:.2f}")
print(f" - Pearson Corr: {corr:.2f}")
print(f"✅ Within ±2 kcal/mol: {within_2:.1f}%")
print(f"✅ Within ±5 kcal/mol: {within_5:.1f}%")
print(f"✅ Within ±10 kcal/mol: {within_10:.1f}%")

# Histogram of ΔG error
plt.figure(figsize=(7, 4))
sns.histplot(df["abs_error"], bins=30, kde=True)
plt.title("Absolute ΔG Error Distribution")
plt.xlabel("|Conditioned ΔG - Computed MFE| (kcal/mol)")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.savefig("../outputs/hist_dg_error.png")
plt.show()

# Scatter: Conditioned vs. Computed
plt.figure(figsize=(6, 6))
sns.scatterplot(x="Conditioned_ΔG_Norm", y="Computed_MFE", data=df, alpha=0.6)
plt.plot([df["Conditioned_ΔG_Norm"].min(), df["Conditioned_ΔG_Norm"].max()],
         [df["Real_ΔG_Target"].min(), df["Real_ΔG_Target"].max()],
         linestyle='--', color='red')
plt.xlabel("Conditioned ΔG (Normalized)")
plt.ylabel("Computed MFE")
plt.title("Conditioned vs. Computed ΔG (MFE)")
plt.grid(True)
plt.tight_layout()
plt.savefig("../outputs/scatter_dg_vs_mfe.png")
plt.show()
