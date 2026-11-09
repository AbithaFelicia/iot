import pandas as pd
from pathlib import Path

base = Path("data/processed/splits")

df = pd.read_csv(
    base / "detector_calibration.csv"
)

# Calculate thresholds using calibration data only
robust_threshold = df["top3_mean_z"].quantile(0.95)

mahal_threshold = df["mahalanobis_squared"].quantile(0.95)

# Calculate calibration false-positive rates
robust_flags = df["top3_mean_z"] > robust_threshold
mahal_flags = df["mahalanobis_squared"] > mahal_threshold

print("========== NEW DETECTOR THRESHOLDS ==========")

print("Calibration records:", len(df))

print("\nROBUST Z-SCORE")
print("Threshold:", robust_threshold)
print("False positives:", robust_flags.sum())
print("FPR:", round(robust_flags.mean() * 100, 4), "%")

print("\nMAHALANOBIS")
print("Threshold:", mahal_threshold)
print("False positives:", mahal_flags.sum())
print("FPR:", round(mahal_flags.mean() * 100, 4), "%")

# Save thresholds
(base / "robust_threshold_new.txt").write_text(
    str(robust_threshold)
)

(base / "mahalanobis_threshold_new.txt").write_text(
    str(mahal_threshold)
)

print("\nNew thresholds saved.")
