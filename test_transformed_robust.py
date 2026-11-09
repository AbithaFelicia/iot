import pandas as pd
import numpy as np
from pathlib import Path

base = Path("data/processed/splits")

print("Loading datasets...")

train = pd.read_csv(base / "train.csv")
cal = pd.read_csv(base / "calibration.csv")

features = train.columns.tolist()

# Transform only the three highly skewed features
transform_features = ["Rate", "Min", "Variance"]

for col in transform_features:
    train[col] = np.log1p(train[col])
    cal[col] = np.log1p(cal[col])

print("Applied log1p to:", transform_features)

# Build a new baseline using transformed training data
baseline = pd.DataFrame(index=features)

baseline["median"] = train.median()
baseline["MAD"] = (train - baseline["median"]).abs().median()

baseline["safe_scale"] = 1.4826 * baseline["MAD"]
baseline["scale_method"] = "MAD"

# Standard deviation fallback when MAD is zero
zero_scale = baseline["safe_scale"] == 0

baseline.loc[zero_scale, "safe_scale"] = train.std()[zero_scale]
baseline.loc[zero_scale, "scale_method"] = "STD"

if (baseline["safe_scale"] <= 0).any():
    raise ValueError("Invalid feature scale detected.")

baseline.to_csv(base / "baseline_stats_transformed.csv")

# Calculate transformed Robust Z-scores
z = (
    (cal - baseline["median"]).abs()
    .div(baseline["safe_scale"])
)

scores = pd.DataFrame({
    "max_abs_z": z.max(axis=1),
    "mean_abs_z": z.mean(axis=1)
})

scores.to_csv(
    base / "robust_z_scores_transformed.csv",
    index=False
)

print("\n========== TRANSFORMED SCORE SUMMARY ==========")
print(scores.describe().round(3))

print("\n========== SCORE PERCENTILES ==========")

print(
    scores.quantile([0.50, 0.90, 0.95, 0.99, 0.999])
    .round(3)
)

print("\n========== FEATURES CAUSING MAXIMUM SCORES ==========")

print(z.idxmax(axis=1).value_counts().head(15))

print("\nSaved transformed baseline and scores.")
print("Experiment completed.")
