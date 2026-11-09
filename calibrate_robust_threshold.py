import pandas as pd
import numpy as np
from pathlib import Path

base = Path("data/processed/splits")

scores = pd.read_csv(
    base / "robust_score_comparison.csv"
)

score = scores["top3_mean_z"]

threshold = score.quantile(0.95)

predictions = (score > threshold).astype(int)

false_positives = predictions.sum()
total = len(predictions)

fpr = false_positives / total

print("========== ROBUST DETECTOR CALIBRATION ==========")

print("Calibration records:", total)
print("Selected score: Top-3 mean Z-score")
print("Target FPR: 0.05")

print("\nThreshold:", threshold)

print("False positives:", false_positives)
print("True negatives:", total - false_positives)

print("Empirical FPR:", round(fpr, 6))
print("Empirical FPR percentage:", round(fpr * 100, 4), "%")

print("\nThreshold calibration completed.")
