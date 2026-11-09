import pandas as pd
import joblib
from pathlib import Path

base = Path("data/processed/splits")

# Load Robust Z-score baseline
baseline = pd.read_csv(
    base / "baseline_stats_safe.csv"
)

# Load Mahalanobis scaler and model
scaler = joblib.load(
    base / "mahalanobis_scaler.joblib"
)

model = joblib.load(
    base / "mahalanobis_model.joblib"
)

print("========== ROBUST BASELINE ==========")
print("Columns:", list(baseline.columns))
print("Number of features:", len(baseline))

print("\n========== MAHALANOBIS SCALER ==========")
print("Number of features:", scaler.n_features_in_)

if hasattr(scaler, "feature_names_in_"):
    print("Scaler features:")
    print(list(scaler.feature_names_in_))

print("\n========== MAHALANOBIS MODEL ==========")
print("Covariance shape:", model.covariance_.shape)

print("\nModel feature verification completed.")
