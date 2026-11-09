import pandas as pd
from pathlib import Path

train_file = Path("data/processed/splits/train.csv")
baseline_file = Path("data/processed/splits/baseline_stats.csv")

df = pd.read_csv(train_file)
baseline = pd.read_csv(baseline_file, index_col=0)

zero_mad_features = baseline[
    baseline["MAD"] == 0
].index.tolist()

print("========== ZERO-MAD FEATURE ANALYSIS ==========")

for col in zero_mad_features:

    values = df[col]

    print("\nFeature:", col)
    print("Unique values:", values.nunique())
    print("Minimum:", values.min())
    print("Maximum:", values.max())
    print("Standard deviation:", round(values.std(), 6))

    print("Most frequent values:")
    print(
        values.value_counts(normalize=True)
        .head(5)
        .mul(100)
        .round(3)
        .to_string()
    )

print("\nInspection completed.")
