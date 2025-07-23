import torch
import pandas as pd
import RNA
from src.models.bit_diffusion import Unet1D, BitDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 30
channels = 2
timesteps = 1000

model_path = "../src/models/bit_diffusion_unet.pt"
target_path = "../data/processed/target_onehot.pt"
csv_path = "../outputs/all_samples_per_dg.csv"

dg_df = pd.read_csv("../data/processed/vienna_rna_full_features.csv")
dg_min = dg_df["Delta_G_MFE"].min()
dg_max = dg_df["Delta_G_MFE"].max()

dg_bins = torch.linspace(0.1, 0.9, steps=9)

unet = Unet1D(dim=256, seq_len=seq_len, channels=channels, cond_dim=1, target_dim=4)
model = BitDiffusion(unet, timesteps=timesteps).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

target_seq = torch.load(target_path)[0:1].to(device)
bit_to_base = {(0, 0): 'A', (0, 1): 'C', (1, 0): 'G', (1, 1): 'U'}

def mutation_count(gRNA, target):
    return sum(1 for a, b in zip(gRNA, target) if a != b)

def decode_bits(seq_tensor):
    """
    Argmax-based decoding for stable nucleotide prediction.
    """
    seq_tensor = seq_tensor.detach().cpu()
    decoded_seq = []
    for i in range(seq_tensor.size(0)):
        bits = (seq_tensor[i] > 0.5).int()
        decoded_seq.append(bit_to_base[tuple(int(x) for x in bits)])
    return ''.join(decoded_seq)

def create_hairpin(t, g): return t + "UUUUU" + g[::-1]
def compute_mfe(s): return RNA.fold_compound(s).mfe()[1]
def denorm_dg(norm): return norm * (dg_max - dg_min) + dg_min

def generate_all_samples(n_per_bin: int, guidance_scale=2.0, enable_guidance=True):
    rows = []
    with torch.no_grad():
        for dg in dg_bins:
            cond = torch.full((n_per_bin, 1), dg.item()).to(device)
            target = target_seq.repeat(n_per_bin, 1, 1).to(device)
            samples = model.sample((n_per_bin, seq_len, channels), cond, target, device,
                                   guidance_scale=guidance_scale, enable_guidance=enable_guidance)

            for i in range(n_per_bin):
                g = decode_bits(samples[i])
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
                    "Mutation_Count": mutations-1,
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
    generate_all_samples(args.n, guidance_scale=args.guidance, enable_guidance=enable_guidance)
