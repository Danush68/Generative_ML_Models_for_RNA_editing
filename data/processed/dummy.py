import pandas as pd

# Load the dataset
file_path = "vienna_rna_full_features.csv"
df = pd.read_csv(file_path)

# Display basic info
print("🧬 Dataset Shape:", df.shape)
print("\n📋 Columns:\n", df.columns.tolist())
print("\n🔍 Data Types:\n", df.dtypes)
print("\n🧾 Sample Rows:\n", df.head())

# Check for nulls
print("\n❗ Null Values per Column:\n", df.isnull().sum())
