import pandas as pd
import numpy as np
from pathlib import Path

folder = Path("data/raw/CSV/Benign_Final")

files = sorted(folder.glob("*.csv"))

for file in files:
    print("\n==========", file.name, "==========")

    df = pd.read_csv(file)

    for col in df.columns:
        raw = df[col]
        numeric = pd.to_numeric(raw, errors="coerce")

        missing = raw.isna().sum()
        non_numeric = (raw.notna() & numeric.isna()).sum()
        infinite = np.isinf(numeric).sum()

        if missing or non_numeric or infinite:
            print(
                col,
                "| Missing:", missing,
                "| Non-numeric:", non_numeric,
                "| Infinite:", infinite
            )

print("\nInspection completed.")
