import pandas as pd
import numpy as np
import joblib
from pathlib import Path

base = Path("data/processed/splits")

scores = pd.read_csv(base / "mahalanobis_scores.csv")
model = joblib.load(base / "mahalanobis_model.joblib")

values = scores["mahalanobis_squared"]

print("========== MAHALANOBIS SCORE PERCENTILES ==========")

print(
    values.quantile(
        [0.50, 0.90, 0.95, 0.99, 0.999]
    ).round(4)
)

print("\n========== SCORE VALIDATION ==========")

print("Missing scores:", values.isna().sum())
print("Infinite scores:", np.isinf(values).sum())
print("Negative scores:", (values < 0).sum())

print("\n========== COVARIANCE MATRIX CHECK ==========")

covariance = model.covariance_

eigenvalues = np.linalg.eigvalsh(covariance)

print("Matrix shape:", covariance.shape)
print("Minimum eigenvalue:", eigenvalues.min())
print("Maximum eigenvalue:", eigenvalues.max())
print("Condition number:", np.linalg.cond(covariance))
print("Shrinkage:", model.shrinkage_)

print("\n========== TOP 10 MAHALANOBIS SCORES ==========")

print(
    scores.nlargest(10, "mahalanobis_squared")
    .round(4)
    .to_string(index=False)
)

print("\nMahalanobis inspection completed.")
