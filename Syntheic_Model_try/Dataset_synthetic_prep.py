import numpy as np
import pandas as pd
from Bio.Seq import Seq
from ViennaRNA import RNA


def generate_synthetic_data(num_samples=1000, seq_length=50):
    data = []
    for _ in range(num_samples):
        # Generate random gRNA and target sequences
        gRNA = ''.join(np.random.choice(['A', 'U', 'C', 'G'], size=seq_length))
        target = ''.join(np.random.choice(['A', 'U', 'C', 'G'], size=seq_length))

        # Predict secondary structure and free energy using ViennaRNA
        structure, energy = RNA.cofold(gRNA + '&' + target)

        # Simulate editing efficiency/specificity (placeholder)
        editing_efficiency = np.random.uniform(0, 1)
        specificity_score = np.random.uniform(0.5, 1)

        data.append({
            'gRNA': gRNA,
            'target': target,
            'editing_efficiency': editing_efficiency,
            'specificity_score': specificity_score,
            'structure': structure,
            'free_energy': energy
        })
    return pd.DataFrame(data)


# Example
df = generate_synthetic_data(num_samples=100)
df.to_csv('synthetic_data.csv', index=False)

print(df)