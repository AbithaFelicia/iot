import pandas as pd
import numpy as np
from pathlib import Path

folder = Path("data/raw/CSV/Benign_Final")
output = Path("data/processed/benign")

output.mkdir(parents=True, exist_ok=True)

files = sorted(folder.glob("*.csv"))

total_before = 0
total_after = 0

for file in files:

    print("\nProcessing:", file.name)

    df = pd.read_csv(file)

    before = len(df)

    # Replace infinite values with missing values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Remove rows containing missing values
    df = df.dropna()

    # Remove duplicate rows
    df = df.drop_duplicates()

    after = len(df)

    total_before += before
    total_after += after

    output_file = output / file.name

    df.to_csv(output_file, index=False)

    print("Original rows:", before)
    print("Cleaned rows:", after)
    print("Removed rows:", before - after)

print("\n========== SUMMARY ==========")
print("Total original rows:", total_before)
print("Total cleaned rows:", total_after)
print("Total removed rows:", total_before - total_after)
print("Cleaning completed.")
