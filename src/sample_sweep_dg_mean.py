# sample_sweep_dg_mean.py

import torch
import pandas as pd
import matplotlib.pyplot as plt
import RNA
import os
from src.models.bit_diffusion import Unet1D, BitDiffusion
from scipy.stats import pearsonr
import seaborn as sns

# === Config ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 30
channels = 2
timesteps = 1000
num_per_bin = 50

model_path = "../src/models/bit_diffusion_unet.pt"
target_path = "../data/processed/target_onehot.pt"
csv_path = "../outputs/generated_sweep_dg.csv"
scatter_plot = "../outputs/plots/per_bin_stripplot.png"
sample_plot = "../outputs/plots/per_sample_dg_vs_mfe.png"

# === ΔG range from original CSV
dg_df = pd.read_csv("../data/processed/vienna_rna_full_features.csv")
dg_min = dg_df["Delta_G_MFE"].min()
dg_max = dg_df["Delta_G_MFE"].max()

dg_bins = torch.linspace(0.1, 0.9, steps=9)

# === Load model
unet = Unet1D(dim=256, seq_len=seq_len, channels=channels, cond_dim=1, target_dim=4)
model = BitDiffusion(unet, timesteps=timesteps).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Load conditioning target
target_seq = torch.load(target_path)[:num_per_bin].to(device)

bit_to_base = {(0, 0): 'A', (0, 1): 'C', (1, 0): 'G', (1, 1): 'U'}


def decode_onehot_rna(onehot_tensor):
    """Convert one-hot encoded RNA to string"""
    idx_to_base = ['A', 'C', 'G', 'U']
    indices = onehot_tensor.argmax(dim=-1).cpu().tolist()
    return ''.join([idx_to_base[i] for i in indices])

def mutation_count(gRNA, target):
    return int(sum(1 for a, b in zip(gRNA, target) if a != b))

def decode_bits(seq_tensor):
    bits = (seq_tensor > 0.5).int().cpu()
    return ''.join([bit_to_base[tuple(int(x) for x in bits[i])] for i in range(bits.size(0))])

def create_hairpin(t, g): return t + "UUUUU" + g[::-1]
def compute_mfe(s): return RNA.fold_compound(s).mfe()[1]
def denorm_dg(norm): return norm * (dg_max - dg_min) + dg_min

rows = []
with torch.no_grad():
    for dg in dg_bins:
        cond = torch.full((num_per_bin, 1), dg.item()).to(device)
        samples = model.sample((num_per_bin, seq_len, channels), cond, target_seq, device)
        decoded = [decode_bits(seq) for seq in samples]
        mfe_temp = 0
        error_temp = 0
        mut_temp = 0
        for g in decoded:
            hairpin = create_hairpin("CUGACUACAGCAUUGCUCAGUACUGCUGUA", g)
            mfe_temp = mfe_temp + compute_mfe(hairpin)
            mfe_c = compute_mfe(hairpin)
            real_dg = denorm_dg(dg.item())
            error_temp = error_temp + abs(real_dg - mfe_c)
            target_str = 'GACUGAUGACGUAACGAGUCAUGACGACAU'
            mut_temp = mut_temp + mutation_count(g, target_str)
        mfe = mfe_temp/num_per_bin
        error = error_temp/num_per_bin
        mutations = int(mut_temp/num_per_bin)

        print(f"ΔG norm: {dg.item():.2f} → real: {real_dg:.2f} | MFE: {mfe:.2f} | error: {error:.2f} | mutations: {mutations}")

        rows.append({
                "Conditioned_ΔG_Norm": round(dg.item(), 3),
                "Real_ΔG_Target": round(real_dg, 2),
                "Generated_gRNA": g,
                "Hairpin": hairpin,
                "Computed_MFE": round(mfe, 2),
                "Abs_Error_DG_vs_MFE": round(error, 2),
            "Mutation_Count": mutations
            })

# === Save CSV
df = pd.DataFrame(rows)
df.to_csv(csv_path, index=False)
print(f"✅ Saved {len(df)} sequences to {csv_path}")

# === Strip plot: Conditioned vs MFE
plt.figure(figsize=(10, 5))
sns.stripplot(
    x="Real_ΔG_Target",
    y="Computed_MFE",
    data=df,
    jitter=0.2,
    color="blue",
    alpha=0.7
)
plt.title("Conditioned ΔG vs Computed MFE (per sample)")
plt.xlabel("Real Conditioned ΔG (kcal/mol)")
plt.ylabel("Computed MFE (kcal/mol)")
plt.grid(True)
plt.tight_layout()
plt.savefig(scatter_plot)
plt.show()

# === Line+Scatter: Sample index vs ΔG & MFE
plt.figure(figsize=(10, 5))
x_vals = list(range(len(df)))

plt.plot(x_vals, df["Real_ΔG_Target"], label="Conditioned ΔG", color="green", linewidth=2)
plt.scatter(x_vals, df["Computed_MFE"], label="Computed MFE", color="blue", alpha=0.7)

plt.title("Per-Sample ΔG vs MFE")
plt.xlabel("Sample Index")
plt.ylabel("ΔG / MFE (kcal/mol)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(sample_plot)
plt.show()

# === Pearson Correlation
r, p = pearsonr(df["Real_ΔG_Target"], df["Computed_MFE"])
print(f"📈 Pearson correlation: r = {r:.3f}, p = {p:.3e}")
