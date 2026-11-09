import pandas as pd
from pathlib import Path

train_file = Path("data/processed/splits/calibration.csv")
baseline_file = Path("data/processed/splits/baseline_stats_safe.csv")

df = pd.read_csv(train_file)
baseline = pd.read_csv(baseline_file, index_col=0)

df = df[baseline.index]

# Calculate absolute Robust Z-scores
z = (
    (df - baseline["median"]).abs()
    .div(baseline["safe_scale"])
)

print("========== TOP FEATURES BY EXTREME SCORES ==========")

print(
    z.quantile([0.50, 0.90, 0.95, 0.99, 0.999])
    .T.sort_values(by=0.999, ascending=False)
    .head(15)
    .to_string()
)

print("\n========== FEATURES MOST OFTEN CAUSING MAX SCORE ==========")

max_features = z.idxmax(axis=1)

print(
    max_features.value_counts()
    .head(15)
    .to_string()
)

print("\n========== 10 RECORDS WITH HIGHEST SCORES ==========")

top_indices = z.max(axis=1).nlargest(10).index

for index in top_indices:
    feature = z.loc[index].idxmax()

    print(
        "Row:", index,
        "| Feature:", feature,
        "| Raw value:", df.loc[index, feature],
        "| Z-score:", round(z.loc[index, feature], 3)
    )

print("\nExtreme score inspection completed.")
