import pandas as pd
import numpy as np
from pathlib import Path

attack_root = Path("data/raw/CSV")

total_rows = 0
total_missing = 0
total_infinite = 0
total_non_numeric = 0

print("========== ATTACK DATA QUALITY ==========")

for folder in sorted(attack_root.iterdir()):

    if not folder.is_dir() or folder.name == "Benign_Final":
        continue

    for file in sorted(folder.glob("*.csv")):

        rows = 0
        missing = 0
        infinite = 0
        non_numeric = 0

        for chunk in pd.read_csv(file, chunksize=100000):

            rows += len(chunk)

            numeric = chunk.apply(
                pd.to_numeric,
                errors="coerce"
            )

            non_numeric += (
                numeric.isna() & chunk.notna()
            ).sum().sum()

            missing += chunk.isna().sum().sum()

            values = numeric.to_numpy(dtype=float)

            infinite += np.isinf(values).sum()

        total_rows += rows
        total_missing += missing
        total_infinite += infinite
        total_non_numeric += non_numeric

        if missing or infinite or non_numeric:
            print(
                f"{folder.name}/{file.name}: "
                f"rows={rows}, missing={missing}, "
                f"infinite={infinite}, "
                f"non_numeric={non_numeric}"
            )

print("\n========== TOTALS ==========")
print("Total attack records:", total_rows)
print("Missing cells:", total_missing)
print("Infinite cells:", total_infinite)
print("Non-numeric cells:", total_non_numeric)

print("\nQuality inspection completed.")
