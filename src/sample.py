# sample.py
# Sampling script with ΔG-only conditioning at inference
# Mutation count signal is ignored (set to neutral)

import torch
import pandas as pd
import RNA
from src.models.bit_diffusion import Unet1D, BitDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 30
channels = 4
timesteps = 1000

model_path = "../src/models/bit_diffusion_unet.pt"
target_path = "../data/processed/target_onehot.pt"
csv_path = "../outputs/all_samples_per_dg.csv"

dg_df = pd.read_csv("../data/processed/vienna_rna_full_features.csv")
dg_min = dg_df["Delta_G_MFE"].min()
dg_max = dg_df["Delta_G_MFE"].max()

dg_bins = torch.linspace(0.1, 0.9, steps=9)

# --- config (only ΔG is active at inference) ---
MUT_DIM = 5          # still exists in model input
COND_DIM = 1 + MUT_DIM

# Model initialization (matches training)
unet = Unet1D(dim=256, seq_len=seq_len, channels=channels,
              cond_dim=COND_DIM, target_dim=4)
model = BitDiffusion(unet, timesteps=timesteps).to(device)

# --- load checkpoint with key remap (no retraining needed) ---
sd = torch.load(model_path, map_location=device)
if any(k.startswith("model.cond_embed.") for k in sd.keys()):
    remapped = {k.replace("model.cond_embed.", "model.dg_embed."): v for k, v in sd.items()}
    sd = remapped
    print("✅ Remapped checkpoint keys: cond_embed → dg_embed")
model.load_state_dict(sd, strict=True)
model.eval()

# Load target sequence (one-hot)
target_seq = torch.load(target_path)[0:1].to(device)

BASES = ['A', 'C', 'G', 'U']

def decode_bits(seq_tensor):
    t = seq_tensor.detach().cpu()
    if t.ndim != 2:
        raise ValueError(f"Expected 2D tensor per sequence, got shape {tuple(t.shape)}")
    if t.shape[1] not in (2, 4) and t.shape[0] in (2, 4):
        t = t.T
    C = t.shape[1]
    if C == 4:
        idx = t.argmax(dim=1).tolist()
        return ''.join(BASES[i] for i in idx)
    if C == 2:
        bits = (t > 0.5).to(torch.int32)
        mapping = {(0,0):'A',(0,1):'C',(1,0):'G',(1,1):'U'}
        return ''.join(mapping.get((int(bits[i,0]), int(bits[i,1])), 'N') for i in range(t.shape[0]))
    raise ValueError(f"Unsupported channel count for decoding: {C}")

def create_hairpin(t, g): return t + "UUUUU" + g[::-1]
def compute_mfe(s): return RNA.fold_compound(s).mfe()[1]
def denorm_dg(norm): return norm * (dg_max - dg_min) + dg_min

def mutation_count_str(gRNA, ref):
    return sum(1 for a, b in zip(gRNA, ref) if a != b)

def generate_all_samples(n_per_bin: int, guidance_scale=2.0):
    rows = []
    # mutation part is always zeros -> ignored
    mut_onehot = torch.zeros(MUT_DIM, dtype=torch.float32, device=device)

    grad_ctx = torch.enable_grad()
    with grad_ctx:
        for dg in dg_bins:
            cond_vec = torch.cat([torch.tensor([dg.item()], device=device), mut_onehot]).unsqueeze(0)
            cond_vec = cond_vec.repeat(n_per_bin, 1)
            target = target_seq.repeat(n_per_bin, 1, 1).to(device)

            samples = model.sample(
                (n_per_bin, seq_len, channels),
                cond_vec, target, device,
                guidance_scale=guidance_scale,
            )

            for i in range(n_per_bin):
                g = decode_bits(samples[i])
                hairpin = create_hairpin("CUGACUACAGCAUUGCUCAGUACUGCUGUA", g)
                mfe = compute_mfe(hairpin)
                real_dg = denorm_dg(dg.item())
                error = abs(real_dg - mfe)
                guide_rna = 'GACUGAUGACGUAACGAGUCAUGACGACAU'
                mcount = mutation_count_str(g, guide_rna)

                rows.append({
                    "Conditioned_ΔG_Norm": round(dg.item(), 3),
                    "Real_ΔG_Target": round(real_dg, 2),
                    "Generated_gRNA": g,
                    "Computed_MFE": round(mfe, 2),
                    "Abs_Error_DG_vs_MFE": round(error, 2),
                    "Achieved_Mutation_Count": mcount,
                    "Hairpin": hairpin
                })

            print(f"✅ Generated {n_per_bin} samples for ΔG norm {dg.item():.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved {len(df)} samples to {csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of samples per ΔG bin")
    parser.add_argument("--guidance", type=float, default=5.0, help="ΔG guidance scale")
    args = parser.parse_args()
    generate_all_samples(args.n, guidance_scale=args.guidance)
