import pandas as pd
from pathlib import Path

base = Path("data/processed/splits")

# Load independent validation data
df = pd.read_csv(base / "benign_validation.csv")

# Load calibrated thresholds
robust_threshold = float(
    (base / "robust_threshold_new.txt").read_text()
)

mahal_threshold = float(
    (base / "mahalanobis_threshold_new.txt").read_text()
)

# Apply fixed thresholds
robust_flags = df["top3_mean_z"] > robust_threshold
mahal_flags = df["mahalanobis_squared"] > mahal_threshold

# Calculate false-positive rates
robust_fp = robust_flags.sum()
mahal_fp = mahal_flags.sum()

total = len(df)

robust_fpr = robust_fp / total
mahal_fpr = mahal_fp / total

print("========== INDEPENDENT BENIGN VALIDATION ==========")

print("Validation records:", total)

print("\nROBUST Z-SCORE")
print("Threshold:", robust_threshold)
print("False positives:", robust_fp)
print("True negatives:", total - robust_fp)
print("Validation FPR:", round(robust_fpr * 100, 4), "%")

print("\nMAHALANOBIS")
print("Threshold:", mahal_threshold)
print("False positives:", mahal_fp)
print("True negatives:", total - mahal_fp)
print("Validation FPR:", round(mahal_fpr * 100, 4), "%")

print("\nValidation completed.")
