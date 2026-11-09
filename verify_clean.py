import pandas as pd
import numpy as np
from pathlib import Path

folder = Path("data/processed/benign")

for file in sorted(folder.glob("*.csv")):

    df = pd.read_csv(file)

    missing = df.isna().sum().sum()

    infinite = np.isinf(
        df.select_dtypes(include=np.number)
    ).sum().sum()

    duplicates = df.duplicated().sum()

    print("\nFile:", file.name)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Missing values:", missing)
    print("Infinite values:", infinite)
    print("Duplicate rows:", duplicates)

print("\nVerification completed.")
