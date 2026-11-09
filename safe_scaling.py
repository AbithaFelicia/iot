import pandas as pd
from pathlib import Path

train_file = Path("data/processed/splits/train.csv")
output_file = Path("data/processed/splits/baseline_stats_safe.csv")

print("Loading training data...")

df = pd.read_csv(train_file)

median = df.median()
mad = (df - median).abs().median()
std = df.std()

robust_scale = 1.4826 * mad

# Use standard deviation when MAD is zero
safe_scale = robust_scale.where(
    robust_scale > 0,
    std
)

baseline = pd.DataFrame({
    "median": median,
    "MAD": mad,
    "std": std,
    "safe_scale": safe_scale
})

baseline["scale_method"] = [
    "MAD" if mad[col] > 0 else "STD_FALLBACK"
    for col in df.columns
]

baseline.to_csv(output_file)

print("\n========== SCALING SUMMARY ==========")
print(baseline[["MAD", "std", "safe_scale", "scale_method"]].to_string())

print("\nZero or invalid scales:")
print(baseline[
    baseline["safe_scale"].isna() |
    (baseline["safe_scale"] <= 0)
])

print("\nFallback features:")
print(baseline[
    baseline["scale_method"] == "STD_FALLBACK"
].index.tolist())

print("\nSaved:", output_file)
