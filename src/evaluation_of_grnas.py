import RNA
import pandas as pd
import matplotlib.pyplot as plt

def string_rev_temp(seq):
    return seq[::-1]

def create_hairpin_temp(original, complimentary):
    loop_sequence = " UUUUU "
    compli = string_rev_temp(complimentary)
    return original + loop_sequence + compli


# Load generated sequences from file
sequences = []
original_rna = "CUGACUACAGCAUUGCUCAGUACUGCUGUA"
with open("generated_sequences.txt", "r") as f:
    current_seq = ""
    for line in f:
        if line.startswith(">"):
            if current_seq:
                sequences.append(current_seq)
                current_seq = ""
        else:
            current_seq += line.strip()
    if current_seq:
        sequences.append(current_seq)

sequences_new = []
for grna in sequences:
    x = create_hairpin_temp(original_rna, grna)
    #print(x)
    sequences_new.append(x)

#print(sequences_new)
# Containers for results
mfe_structures = []
delta_gs = []

for seq in sequences_new:
    fc = RNA.fold_compound(seq)
    struct, mfe = fc.mfe()
    mfe_structures.append(struct)
    delta_gs.append(mfe)

# Save to CSV
df = pd.DataFrame({
    "Sequence": sequences_new,
    "gRNA": sequences,
    "MFE Structure": mfe_structures,
    "Delta_G_MFE": delta_gs
})

df.to_csv("generated_evaluated.csv", index=False)
print("✅ Evaluation complete. Saved to 'generated_evaluated.csv'")

# Plot ΔG values as a line plot
plt.figure(figsize=(8, 4))
plt.plot(range(len(delta_gs)), delta_gs, marker='o', linestyle='-', color='blue')
plt.title("Delta G (MFE) Values for Generated gRNAs")
plt.xlabel("Sequence Index")
plt.ylabel("Delta G (kcal/mol)")
plt.grid(True)
plt.tight_layout()
plt.savefig("../outputs/plots/delta_g_lineplot.png")
plt.show()
