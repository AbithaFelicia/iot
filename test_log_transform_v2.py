import pandas as pd
import numpy as np
from pathlib import Path

base = Path("data/processed/splits")

print("Loading datasets...")

train = pd.read_csv(base / "train.csv")
cal = pd.read_csv(base / "calibration.csv")

features = train.columns.tolist()

log_features = [
    "Rate",
    "Min",
    "Variance",
    "Tot sum"
]

# Apply the same transformations to training and calibration
for col in log_features:
    if train[col].min() < 0 or cal[col].min() < 0:
        raise ValueError(f"Negative values found in {col}")

    train[col] = np.log1p(train[col])
    cal[col] = np.log1p(cal[col])

print("Applied log1p to:", log_features)

# Calculate baseline using training data only
median = train.median()
mad = (train - median).abs().median()
scale = 1.4826 * mad

# Use standard deviation when MAD is zero
std = train.std()
scale = scale.where(scale > 0, std)

if (scale <= 0).any():
    raise ValueError("Some features have zero scaling values.")

# Calculate calibration scores
z = (cal - median).abs().div(scale)

max_z = z.max(axis=1)
mean_z = z.mean(axis=1)

print("\n========== NEW TRANSFORMED SCORE SUMMARY ==========")

scores = pd.DataFrame({
    "max_abs_z": max_z,
    "mean_abs_z": mean_z
})

print(scores.describe().round(3))

print("\n========== SCORE PERCENTILES ==========")

print(
    scores.quantile([0.50, 0.90, 0.95, 0.99, 0.999])
    .round(3)
)

print("\n========== FEATURES CAUSING MAXIMUM SCORES ==========")

print(
    z.idxmax(axis=1)
    .value_counts()
    .head(15)
)

print("\n========== TOP FEATURE EXTREMES ==========")

print(
    z.quantile([0.95, 0.99, 0.999])
    .T.sort_values(by=0.999, ascending=False)
    .head(10)
    .round(3)
)

# Save separate experimental outputs
scores.to_csv(base / "robust_z_scores_log_v2.csv")

print("\nSaved: robust_z_scores_log_v2.csv")
print("Experiment completed.")
