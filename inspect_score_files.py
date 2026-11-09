import pandas as pd
from pathlib import Path

base = Path("data/processed/splits")

files = [
    "robust_score_comparison.csv",
    "mahalanobis_scores.csv",
    "detector_agreement.csv"
]

for filename in files:
    path = base / filename

    print("\n==========", filename, "==========")

    df = pd.read_csv(path)

    print("Records:", len(df))
    print("Columns:", list(df.columns))
    print("First 3 rows:")
    print(df.head(3).to_string(index=False))

print("\nInspection completed.")
