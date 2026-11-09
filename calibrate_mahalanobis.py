import pandas as pd
from pathlib import Path

base = Path("data/processed/splits")

scores = pd.read_csv(
    base / "mahalanobis_scores.csv"
)

values = scores["mahalanobis_squared"]

# Calculate threshold from benign calibration data
threshold = values.quantile(0.95)

# Flag records above the threshold
predictions = (values > threshold).astype(int)

false_positives = predictions.sum()
total = len(predictions)

fpr = false_positives / total

print("========== MAHALANOBIS CALIBRATION ==========")

print("Calibration records:", total)
print("Target FPR: 5%")

print("\nThreshold:", threshold)
print("False positives:", false_positives)
print("True negatives:", total - false_positives)

print("Empirical FPR:", round(fpr, 6))
print("Empirical FPR percentage:", round(fpr * 100, 4), "%")

# Save threshold separately
threshold_file = base / "mahalanobis_threshold.txt"

threshold_file.write_text(str(threshold))

print("\nSaved threshold:", threshold_file)
print("Calibration completed.")
