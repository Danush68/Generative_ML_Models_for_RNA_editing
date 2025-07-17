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
num_samples_per_bin = 300

model_path = "../src/models/bit_diffusion_unet.pt"
target_path = "../data/processed/target_onehot.pt"
csv_path = "../outputs/best_sample_per_dg.csv"
plot_path = "../outputs/plots/best_sample_alignment.png"

# Load true ΔG range from dataset
dg_df = pd.read_csv("../data/processed/vienna_rna_full_features.csv")
dg_min = dg_df["Delta_G_MFE"].min()
dg_max = dg_df["Delta_G_MFE"].max()

dg_bins = torch.linspace(0.1, 0.9, steps=9)

unet = Unet1D(dim=256, seq_len=seq_len, channels=channels, cond_dim=1, target_dim=4)
model = BitDiffusion(unet, timesteps=timesteps).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

target_seq = torch.load(target_path)[0:1].repeat(num_samples_per_bin, 1, 1).to(device)

bit_to_base = {(0, 0): 'A', (0, 1): 'C', (1, 0): 'G', (1, 1): 'U'}

def mutation_count(gRNA, target):
    return sum(1 for a, b in zip(gRNA, target) if a != b)

def decode_bits(seq_tensor):
    bits = (seq_tensor > 0.5).int().cpu()
    return ''.join([bit_to_base[tuple(int(x) for x in bits[i])] for i in range(bits.size(0))])

def create_hairpin(t, g): return t + "UUUUU" + g[::-1]
def compute_mfe(s): return RNA.fold_compound(s).mfe()[1]
def denorm_dg(norm): return norm * (dg_max - dg_min) + dg_min

# === Sample and keep best match per DG
rows = []
with torch.no_grad():
    for dg in dg_bins:
        cond = torch.full((num_samples_per_bin, 1), dg.item()).to(device)
        samples = model.sample((num_samples_per_bin, seq_len, channels), cond, target_seq, device)

        best_error = float('inf')
        best_row = None

        for i in range(num_samples_per_bin):
            g = decode_bits(samples[i])
            hairpin = create_hairpin("CUGACUACAGCAUUGCUCAGUACUGCUGUA", g)
            mfe = compute_mfe(hairpin)
            real_dg = denorm_dg(dg.item())
            error = abs(real_dg - mfe)
            target_str = 'GACUGAUGACGUAACGAGUCAUGACGACAU'
            mutations = mutation_count(g, target_str)
            if error < best_error:
                best_row = {
                    "Conditioned_ΔG_Norm": round(dg.item(), 3),
                    "Real_ΔG_Target": round(real_dg, 2),
                    "Generated_gRNA": g,
                    "Computed_MFE": round(mfe, 2),
                    "Abs_Error_DG_vs_MFE": round(error, 2),
                    "Mutation_Count": mutations-1,
                    "Hairpin" : hairpin
                }
                best_error = error
                best_mfe = mfe
                best_error_val = error
                best_mut = mutations
        print(f"ΔG norm: {dg.item():.2f} → real: {real_dg:.2f} | MFE: {best_mfe:.2f} | error: {best_error:.2f} | mutations: {best_mut}")
        rows.append(best_row)

# Save best per bin
df = pd.DataFrame(rows)
df.to_csv(csv_path, index=False)
print(f"✅ Saved best samples to {csv_path}")

# Plot alignment
plt.figure(figsize=(8, 4))
plt.plot(df["Conditioned_ΔG_Norm"], df["Real_ΔG_Target"], label="Conditioned ΔG", marker='o', color="green")
plt.plot(df["Conditioned_ΔG_Norm"], df["Computed_MFE"], label="Best Computed MFE", marker='x', color="blue")
plt.title("Best Sample: Conditioned vs Computed ΔG")
plt.xlabel("Conditioned ΔG (normalized)")
plt.ylabel("ΔG / MFE (kcal/mol)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(plot_path)
plt.show()
