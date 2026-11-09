import pandas as pd

base = "data/processed/splits/"

scores = pd.read_csv(
    base + "robust_z_scores_log_v2.csv"
)

calibration = pd.read_csv(
    base + "calibration.csv"
)

baseline = pd.read_csv(
    base + "baseline_stats_safe.csv",
    index_col=0
)

import numpy as np

features = baseline.index.tolist()

train = pd.read_csv(base + "train.csv")
cal = calibration[features].copy()

train = train[features].copy()

log_features = ["Rate", "Min", "Variance"]

for col in log_features:
    train[col] = np.log1p(train[col])
    cal[col] = np.log1p(cal[col])

median = train.median()
mad = (train - median).abs().median()
scale = 1.4826 * mad
scale = scale.where(scale > 0, train.std())

z = (cal - median).abs().div(scale)

top3_mean = z.apply(
    lambda row: row.nlargest(3).mean(),
    axis=1
)

comparison = pd.DataFrame({
    "max_z": z.max(axis=1),
    "mean_z": z.mean(axis=1),
    "top3_mean_z": top3_mean
})

print("========== SCORE COMPARISON ==========")

print(
    comparison.quantile(
        [0.50, 0.90, 0.95, 0.99, 0.999]
    ).round(3)
)

print("\n========== SCORE CORRELATIONS ==========")

print(comparison.corr().round(4))

print("\n========== TOP 10 TOP-3 SCORES ==========")

print(
    comparison.nlargest(10, "top3_mean_z")
    .round(3)
    .to_string()
)

comparison.to_csv(
    base + "robust_score_comparison.csv",
    index=False
)

print("\nSaved: robust_score_comparison.csv")
print("Comparison completed.")
