#danush
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

# Model initialization (matches training)
unet = Unet1D(dim=256, seq_len=seq_len, channels=channels, cond_dim=1, target_dim=4)
model = BitDiffusion(unet, timesteps=timesteps).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load target sequence (one-hot)
target_seq = torch.load(target_path)[0:1].to(device)

def mutation_count(gRNA, target):
    return sum(1 for a, b in zip(gRNA, target) if a != b)

BASES = ['A', 'C', 'G', 'U']

def decode_bits(seq_tensor):
    """
    Decode a per-position tensor to a base string.
    Supports (L,4) or (4,L) one-hot/probs; also tolerates (L,2) bit-pairs.
    """
    import torch
    t = seq_tensor.detach().cpu()

    if t.ndim != 2:
        raise ValueError(f"Expected 2D tensor per sequence, got shape {tuple(t.shape)}")

    # Normalize shape to (L, C)
    if t.shape[1] not in (2, 4) and t.shape[0] in (2, 4):
        t = t.T
    C = t.shape[1]

    if C == 4:
        # Argmax over A,C,G,U
        idx = t.argmax(dim=1).tolist()
        return ''.join(BASES[i] for i in idx)

    if C == 2:
        # Fallback for bit-pairs (shouldn't be used in your current setup)
        bits = (t > 0.5).to(torch.int32)
        mapping = {(0, 0): 'A', (0, 1): 'C', (1, 0): 'G', (1, 1): 'U'}
        return ''.join(mapping.get((int(bits[i, 0]), int(bits[i, 1])), 'N') for i in range(t.shape[0]))

    raise ValueError(f"Unsupported channel count for decoding: {C} (expected 4 or 2)")


def create_hairpin(t, g): return t + "UUUUU" + g[::-1]
def compute_mfe(s): return RNA.fold_compound(s).mfe()[1]
def denorm_dg(norm): return norm * (dg_max - dg_min) + dg_min

def generate_all_samples(n_per_bin: int, guidance_scale=2.0, enable_guidance=True):
    rows = []

    grad_context = torch.no_grad() if not enable_guidance else torch.enable_grad()
    with grad_context:
        for dg in dg_bins:
            cond = torch.full((n_per_bin, 1), dg.item()).to(device)
            # Keep target same shape as training (batch, seq_len, 4)
            target = target_seq.repeat(n_per_bin, 1, 1).to(device)

            # Call sample with training input shape (batch, seq_len, channels)
            samples = model.sample(
                (n_per_bin, seq_len, channels),
                cond, target, device,
                guidance_scale=guidance_scale,
                enable_guidance=enable_guidance
            )

            for i in range(n_per_bin):
                g = decode_bits(samples[i])  # (seq_len, 2)
                hairpin = create_hairpin("CUGACUACAGCAUUGCUCAGUACUGCUGUA", g)
                mfe = compute_mfe(hairpin)
                real_dg = denorm_dg(dg.item())
                error = abs(real_dg - mfe)
                target_str = 'GACUGAUGACGUAACGAGUCAUGACGACAU'
                mutations = mutation_count(g, target_str)

                rows.append({
                    "Conditioned_ΔG_Norm": round(dg.item(), 3),
                    "Real_ΔG_Target": round(real_dg, 2),
                    "Generated_gRNA": g,
                    "Computed_MFE": round(mfe, 2),
                    "Abs_Error_DG_vs_MFE": round(error, 2),
                    "Mutation_Count": mutations - 1,
                    "Hairpin": hairpin
                })

            print(f"✅ Generated {n_per_bin} samples for ΔG norm {dg.item():.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved {len(df)} samples to {csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of samples per ΔG bin")
    parser.add_argument("--guidance", type=float, default=2.0, help="ΔG guidance scale")
    parser.add_argument("--no-guidance", action="store_true", help="Disable ΔG guidance")
    args = parser.parse_args()

    enable_guidance = not args.no_guidance
    generate_all_samples(args.n, guidance_scale=args.guidance, enable_guidance=True)
