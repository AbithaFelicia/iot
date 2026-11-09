import pandas as pd
from pathlib import Path

base = Path("data/processed/splits")

robust = pd.read_csv(base / "robust_score_comparison.csv")
mahal = pd.read_csv(base / "mahalanobis_scores.csv")

# Verify matching records
assert len(robust) == len(mahal)

# Fixed random seed for reproducibility
df = pd.concat([robust, mahal], axis=1)

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Split into two equal groups
mid = len(df) // 2

calibration = df.iloc[:mid].copy()
validation = df.iloc[mid:].copy()

# Save both groups
calibration.to_csv(
    base / "detector_calibration.csv",
    index=False
)

validation.to_csv(
    base / "benign_validation.csv",
    index=False
)

print("========== DATA SPLIT ==========")

print("Original records:", len(df))
print("Calibration records:", len(calibration))
print("Validation records:", len(validation))

print("\nCalibration columns:", list(calibration.columns))
print("Validation columns:", list(validation.columns))

print("\nFiles saved successfully.")
