import pandas as pd
import numpy as np
import joblib
from pathlib import Path

base = Path("data/processed/splits")
attack_root = Path("data/raw/CSV")

scaler = joblib.load(
    base / "mahalanobis_scaler.joblib"
)

features = list(scaler.feature_names_in_)

log_features = ["Rate", "Min", "Variance"]

total_rows = 0
invalid_rows = 0
invalid_cells = 0

feature_invalid = {
    feature: 0 for feature in features
}

print("========== SELECTED FEATURE QUALITY ==========")

for folder in sorted(attack_root.iterdir()):

    if not folder.is_dir() or folder.name == "Benign_Final":
        continue

    for file in sorted(folder.glob("*.csv")):

        file_rows = 0
        file_invalid_rows = 0

        for chunk in pd.read_csv(
            file,
            usecols=features,
            chunksize=100000
        ):

            file_rows += len(chunk)
            total_rows += len(chunk)

            numeric = chunk.apply(
                pd.to_numeric,
                errors="coerce"
            )

            values = numeric.to_numpy(dtype=float)

            invalid_mask = (
                numeric.isna().to_numpy()
                | ~np.isfinite(values)
            )

            # Check log1p transformation validity
            for feature in log_features:
                col_index = features.index(feature)

                transformed = np.log1p(
                    values[:, col_index],
                    where=values[:, col_index] > -1,
                    out=np.full(len(chunk), np.nan)
                )

                invalid_mask[:, col_index] |= (
                    ~np.isfinite(transformed)
                )

            row_mask = invalid_mask.any(axis=1)

            file_invalid_rows += row_mask.sum()
            invalid_rows += row_mask.sum()

            invalid_cells += invalid_mask.sum()

            for i, feature in enumerate(features):
                feature_invalid[feature] += (
                    invalid_mask[:, i].sum()
                )

        if file_invalid_rows:
            print(
                f"{folder.name}/{file.name}: "
                f"rows={file_rows}, "
                f"invalid rows={file_invalid_rows}"
            )

print("\n========== FINAL QUALITY REPORT ==========")

print("Total attack records:", total_rows)
print("Records with invalid selected features:", invalid_rows)
print("Records with valid selected features:", total_rows - invalid_rows)
print("Invalid feature cells:", invalid_cells)

print("\n========== INVALID VALUES BY FEATURE ==========")

for feature, count in feature_invalid.items():
    if count:
        print(feature, ":", count)

print("\nInspection completed.")
