import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

base = Path("data/processed/splits")

print("Loading benign training data...")

train = pd.read_csv(base / "train.csv")
cal = pd.read_csv(base / "calibration.csv")

# Apply the same transformations used by Robust Z-score
log_features = ["Rate", "Min", "Variance"]

for col in log_features:
    if train[col].min() < 0 or cal[col].min() < 0:
        raise ValueError(f"Negative values found in {col}")

    train[col] = np.log1p(train[col])
    cal[col] = np.log1p(cal[col])

print("Applied log1p to:", log_features)

# Standardize using training data only
print("\nStandardizing features...")

scaler = StandardScaler()

X_train = scaler.fit_transform(train)
X_cal = scaler.transform(cal)

# Fit regularized covariance model using benign training data
print("\nFitting Ledoit-Wolf covariance model...")

model = LedoitWolf()
model.fit(X_train)

print("Covariance model fitted.")

# Calculate squared Mahalanobis distances
print("\nCalculating calibration scores...")

mahal_scores = model.mahalanobis(X_cal)

results = pd.DataFrame({
    "mahalanobis_squared": mahal_scores
})

# Save calibration scores
output_file = base / "mahalanobis_scores.csv"
results.to_csv(output_file, index=False)

# Save fitted model and scaler for later testing
joblib.dump(scaler, base / "mahalanobis_scaler.joblib")
joblib.dump(model, base / "mahalanobis_model.joblib")

print("\n========== MAHALANOBIS SUMMARY ==========")

print(results.describe().round(4))

print("\n========== MODEL INFORMATION ==========")

print("Training records:", len(X_train))
print("Calibration records:", len(X_cal))
print("Features:", X_train.shape[1])
print("Ledoit-Wolf shrinkage:", round(model.shrinkage_, 6))
print(
    "Covariance condition number:",
    round(np.linalg.cond(model.covariance_), 4)
)

print("\nSaved:", output_file)
print("Saved: mahalanobis_scaler.joblib")
print("Saved: mahalanobis_model.joblib")

print("\nMahalanobis training completed.")
