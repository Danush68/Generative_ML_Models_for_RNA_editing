import numpy as np
import pandas as pd

# Load latent vectors
latents = np.load("../features/latent_vectors.npy")

# Convert to DataFrame and save as CSV
df = pd.DataFrame(latents)
df.to_csv("../features/latent_vectors.csv", index=False)
print("Saved latent vectors as CSV.")
