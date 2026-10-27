import pandas as pd
from pathlib import Path

path = Path("data/raw/unsw_nb15/UNSW_NB15_training-set.csv")

df = pd.read_csv(path)

print("\n=== UNSW-NB15 DATA AUDIT ===")

print("\n1. Dataset dimensions:")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\n2. Column names:")
print(df.columns.tolist())

print("\n3. Class distribution:")
print(df["label"].value_counts(dropna=False))

print("\n4. Class percentages:")
print(df["label"].value_counts(normalize=True, dropna=False) * 100)

print("\n5. Missing values per column:")
print(df.isna().sum().to_string())

print("\n6. Missing target labels:")
print(df["label"].isna().sum())

print("\n7. Exact duplicate records:")
print(df.duplicated().sum())

print("\n8. Available attack categories:")
if "attack_cat" in df.columns:
    print(df["attack_cat"].value_counts(dropna=False))

print("\n9.Duplicate records excluding ID:",)
print(
    df.drop(columns=["id"]).duplicated().sum()
)

