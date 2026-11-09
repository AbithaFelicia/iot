import pandas as pd
from pathlib import Path

base = Path("data/processed/splits")

baseline = pd.read_csv(
    base / "baseline_stats_safe.csv",
    index_col=0
)

scaler = __import__("joblib").load(
    base / "mahalanobis_scaler.joblib"
)

robust_features = baseline.index.tolist()
mahal_features = list(scaler.feature_names_in_)

print("Robust feature count:", len(robust_features))
print("Mahalanobis feature count:", len(mahal_features))

print("\nFeature order matches:",
      robust_features == mahal_features)

if robust_features != mahal_features:
    print("\nRobust-only/order differences:")
    for i, (a, b) in enumerate(
        zip(robust_features, mahal_features)
    ):
        if a != b:
            print(i, a, "!=", b)

print("\nFeature verification completed.")
