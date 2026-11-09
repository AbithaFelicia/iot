import pandas as pd
from pathlib import Path

calibration_file = Path("data/processed/splits/calibration.csv")
baseline_file = Path("data/processed/splits/baseline_stats_safe.csv")

output_file = Path("data/processed/splits/robust_z_scores.csv")

print("Loading calibration dataset...")

df = pd.read_csv(calibration_file)
baseline = pd.read_csv(baseline_file, index_col=0)

# Ensure the feature order matches the baseline
df = df[baseline.index]

print("Calibration records:", len(df))
print("Features:", len(df.columns))

# Calculate absolute Robust Z-scores
z_scores = (
    (df - baseline["median"]).abs()
    .div(baseline["safe_scale"])
)

# Calculate two anomaly score summaries
results = pd.DataFrame({
    "max_abs_z": z_scores.max(axis=1),
    "mean_abs_z": z_scores.mean(axis=1)
})

results.to_csv(output_file, index=False)

print("\n========== ROBUST Z-SCORE SUMMARY ==========")
print(results.describe().to_string())

print("\nScore percentiles:")
print(
    results["max_abs_z"]
    .quantile([0.50, 0.90, 0.95, 0.99, 0.999])
    .to_string()
)

print("\nSaved:", output_file)
print("Robust Z-score calculation completed.")
