import pandas as pd
import numpy as np
from pathlib import Path

folder = Path("data/raw/CSV/Benign_Final")

files = sorted(folder.glob("*.csv"))

total_rows = 0

for file in files:
    print("\nChecking:", file.name)

    df = pd.read_csv(file)

    rows = len(df)
    total_rows += rows

    print("Rows:", rows)
    print("Columns:", len(df.columns))

    numeric = df.apply(pd.to_numeric, errors="coerce")

    missing = numeric.isna().sum().sum()

    infinite = np.isinf(
        numeric.select_dtypes(include=np.number)
    ).sum().sum()

    duplicates = df.duplicated().sum()

    print("Missing/non-numeric cells:", missing)
    print("Infinite values:", infinite)
    print("Duplicate rows:", duplicates)

print("\nTotal benign records:", total_rows)
print("Validation completed.")
