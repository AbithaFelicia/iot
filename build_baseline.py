import pandas as pd
from pathlib import Path

train_file = Path("data/processed/splits/train.csv")
output_file = Path("data/processed/splits/baseline_stats.csv")

print("Loading training dataset...")

df = pd.read_csv(train_file)

print("Training records:", len(df))
print("Features:", len(df.columns))

# Calculate median for each feature
median = df.median()

# Calculate Median Absolute Deviation (MAD)
mad = (df - median).abs().median()

# Calculate robust scale
robust_scale = 1.4826 * mad

# Create baseline statistics
baseline = pd.DataFrame({
    "median": median,
    "MAD": mad,
    "robust_scale": robust_scale
})

baseline.to_csv(output_file)

print("\n========== BASELINE SUMMARY ==========")
print(baseline.to_string())

# Identify features with zero MAD
zero_mad = baseline[baseline["MAD"] == 0].index.tolist()

print("\nFeatures with zero MAD:")
print(zero_mad if zero_mad else "None")

print("\nBaseline saved to:", output_file)
print("Baseline calculation completed.")
