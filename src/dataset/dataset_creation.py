import random
import pandas as pd
from tqdm import tqdm

import Combining_with_haipin as Ch
import time

#Fixed original RNA sequence (centered around 2019th mutation)
original_rna = "CTGACTACAGCATTGCTCAGTACTGCTGTA"
complementary_rna = Ch.complement_rna(original_rna)

#Generate 1,000,000 mutated samples with random 1–10 mutations in each mutated grna
samples = []
num_samples = 1000000

complementary_rna_1 = list(complementary_rna)
complementary_rna_1[8] = 'A' # for the mutation
complementary_rna_1 = ''.join(complementary_rna_1)
complementary_rna = complementary_rna_1

for _ in range(num_samples):
    num_mutations = random.randint(1, 7)
    mutated_compl = Ch.mutate_rna_sequence(complementary_rna, num_mutations)
    mutated_hairpin = Ch.create_hairpin(original_rna, mutated_compl)
    samples.append({
        "Original_RNA": original_rna,
        "Guide RNA": complementary_rna,
        "Mutated_gRNA": mutated_compl,
        "Mutations": num_mutations,
        "Hairpin_RNA": mutated_hairpin,
    })

#Save to CSV
df = pd.DataFrame(samples)
df.to_csv("../../data/raw/hairpin_rna_random_mutations.csv", index=False)

for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Saving Data"):
    # Simulate some processing time
    time.sleep(0.0001)  # Adjust the sleep time as needed
print("✅ Saved 1,000,000 hairpin RNAs with 1–10 mutations to 'hairpin_rna_random_mutations.csv'")
