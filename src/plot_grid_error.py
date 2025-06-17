# plot_dg_error_grid.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load sweep results
df = pd.read_csv("../outputs/generated_sweep_dg.csv")

# === Optional: sort for cleaner grid
df = df.sort_values("Conditioned_ΔG_Norm")

# === Plot grid: ΔG vs Absolute Error
plt.figure(figsize=(10, 5))

sns.stripplot(
    x="Conditioned_ΔG_Norm",
    y="Abs_Error_DG_vs_MFE",
    data=df,
    jitter=0.2,
    size=5,
    alpha=0.7,
    color="purple"
)

plt.title("Absolute Error vs Conditioned ΔG")
plt.xlabel("Conditioned ΔG (normalized)")
plt.ylabel("Absolute Error: |Conditioned ΔG − Computed MFE| (kcal/mol)")
plt.grid(True)
plt.tight_layout()

# === Save
os.makedirs("../outputs/plots", exist_ok=True)
out_path = "../outputs/plots/error_vs_dg_stripplot.png"
plt.savefig(out_path)
plt.show()

print(f"✅ Saved error plot to {out_path}")
