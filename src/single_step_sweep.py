# sample_single_per_step.py

import torch
import pandas as pd
import matplotlib.pyplot as plt
import RNA
import os
from src.models.bit_diffusion import Unet1D, BitDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 30
channels = 2
timesteps = 1000

model_path = "../src/models/bit_diffusion_unet.pt"
target_path = "../data/processed/target_onehot.pt"
out_csv = "../outputs/single_per_step.csv"
out_plot = "../outputs/plots/single_step_plot.png"

# ✅ NEW: Load real ΔG range
dg_df = pd.read_csv("../data/processed/vienna_rna_full_features.csv")
dg_min = dg_df["Delta_G_MFE"].min()
dg_max = dg_df["Delta_G_MFE"].max()

dg_steps = torch.linspace(0.1, 0.9, steps=9)

unet = Unet1D(dim=256, seq_len=seq_len, channels=channels, cond_dim=1, target_dim=4)
model = BitDiffusion(unet, timesteps=timesteps).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

target_seq = torch.load(target_path)[0:1].to(device)

bit_to_base = {(0, 0): 'A', (0, 1): 'C', (1, 0): 'G', (1, 1): 'U'}


def decode_onehot_rna(onehot_tensor):
    """Convert one-hot encoded RNA to string"""
    idx_to_base = ['A', 'C', 'G', 'U']
    indices = onehot_tensor.argmax(dim=-1).cpu().tolist()
    return ''.join([idx_to_base[i] for i in indices])

def mutation_count(gRNA, target):
    return sum(1 for a, b in zip(gRNA, target) if a != b)

def decode_bits(seq_tensor):
    bits = (seq_tensor > 0.5).int().cpu()
    return ''.join([bit_to_base[tuple(int(x) for x in bits[i])] for i in range(bits.size(0))])
def create_hairpin(t, g): return t + "UUUUU" + g[::-1]
def compute_mfe(s): return RNA.fold_compound(s).mfe()[1]
def denorm_dg(norm): return norm * (dg_max - dg_min) + dg_min

rows = []
with torch.no_grad():
    for dg in dg_steps:
        cond = torch.tensor([[dg.item()]], device=device)
        sample = model.sample((1, seq_len, channels), cond, target_seq, device)[0]
        g = decode_bits(sample)
        hairpin = create_hairpin("CUGACUACAGCAUUGCUCAGUACUGCUGUA", g)
        mfe = compute_mfe(hairpin)
        real_dg = denorm_dg(dg.item())
        error = abs(real_dg - mfe)

        print(f"ΔG norm: {dg.item():.2f} → real: {real_dg:.2f} | MFE: {mfe:.2f} | error: {error:.2f}")

        target_str = 'GACUGAUGACGUAACGAGUCAUGACGACAU'
        mutations = mutation_count(g, target_str)
        rows.append({
            "Conditioned_ΔG_Norm": round(dg.item(), 3),
            "Real_ΔG_Target": round(real_dg, 2),
            "Generated_gRNA": g,
            "Computed_MFE": round(mfe, 2),
            "Abs_Error_DG_vs_MFE": round(error, 2),
            "Mutation_Count": mutations
        })

df = pd.DataFrame(rows)
os.makedirs("outputs/plots", exist_ok=True)
df.to_csv(out_csv, index=False)

# === Updated Plot: Conditioned vs Computed MFE over DG steps
plt.figure(figsize=(8, 4))

plt.plot(df["Conditioned_ΔG_Norm"], df["Real_ΔG_Target"], label="Conditioned ΔG (real)", color="green", marker='o')
plt.plot(df["Conditioned_ΔG_Norm"], df["Computed_MFE"], label="Computed MFE", color="blue", marker='x')

plt.title("Conditioned ΔG vs Computed MFE Across Steps")
plt.xlabel("Conditioned ΔG (normalized)")
plt.ylabel("ΔG / MFE (kcal/mol)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(out_plot)
plt.show()

