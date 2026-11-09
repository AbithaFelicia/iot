
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

BASE = Path("data/processed/splits")
ATTACK_ROOT = Path("data/raw/CSV")
CHUNK_SIZE = 100000

LOG_FEATURES = ["Rate", "Min", "Variance"]

# Load the correctly transformed Robust baseline
baseline = pd.read_csv(
    BASE / "baseline_stats_transformed.csv",
    index_col=0
)

# Load the trained Mahalanobis model and scaler
scaler = joblib.load(BASE / "mahalanobis_scaler.joblib")
model = joblib.load(BASE / "mahalanobis_model.joblib")

features = list(scaler.feature_names_in_)

# Verify feature consistency
if set(features) != set(baseline.index):
    raise ValueError("Feature mismatch between baseline and scaler.")

baseline = baseline.loc[features]

median = baseline["median"].to_numpy()
scale = baseline["safe_scale"].to_numpy()

# Load fixed thresholds
robust_threshold = float(
    (BASE / "robust_threshold_new.txt").read_text().strip()
)

mahal_threshold = float(
    (BASE / "mahalanobis_threshold_new.txt").read_text().strip()
)

print("========== ATTACK EVALUATION ==========")
print("Features:", len(features))
print("Robust threshold:", robust_threshold)
print("Mahalanobis threshold:", mahal_threshold)

results = []

total_rows = 0
total_valid = 0
total_invalid = 0
total_robust = 0
total_mahal = 0
total_both = 0

# Process each attack category
for folder in sorted(ATTACK_ROOT.iterdir()):

    if not folder.is_dir() or folder.name == "Benign_Final":
        continue

    files = sorted(folder.glob("*.csv"))

    if not files:
        continue

    category_rows = 0
    category_valid = 0
    category_invalid = 0
    category_robust = 0
    category_mahal = 0
    category_both = 0

    for file in files:

        for chunk in pd.read_csv(
            file,
            usecols=features,
            chunksize=CHUNK_SIZE
        ):

            category_rows += len(chunk)
            total_rows += len(chunk)

            numeric = chunk.apply(
                pd.to_numeric,
                errors="coerce"
            )

            values = numeric.to_numpy(dtype=float)

            valid_mask = (
                numeric.notna().all(axis=1).to_numpy()
                & np.isfinite(values).all(axis=1)
            )

            # Check log1p domain
            for feature in LOG_FEATURES:
                index = features.index(feature)
                valid_mask &= values[:, index] > -1

            invalid_count = int((~valid_mask).sum())

            category_invalid += invalid_count
            total_invalid += invalid_count

            if not valid_mask.any():
                continue

            valid_values = values[valid_mask].copy()

            # Apply the same transformation used during training
            for feature in LOG_FEATURES:
                index = features.index(feature)
                valid_values[:, index] = np.log1p(
                    valid_values[:, index]
                )

            # Robust Z-score
            z = np.abs(
                (valid_values - median) / scale
            )

            robust_scores = np.sort(
                z,
                axis=1
            )[:, -3:].mean(axis=1)

            robust_flags = (
                robust_scores > robust_threshold
            )

            # Mahalanobis distance
            scaled_values = scaler.transform(valid_values)

            mahal_scores = model.mahalanobis(
                scaled_values
            )

            mahal_flags = (
                mahal_scores > mahal_threshold
            )

            both_flags = robust_flags & mahal_flags

            count = len(valid_values)

            category_valid += count
            total_valid += count

            category_robust += int(robust_flags.sum())
            category_mahal += int(mahal_flags.sum())
            category_both += int(both_flags.sum())

            total_robust += int(robust_flags.sum())
            total_mahal += int(mahal_flags.sum())
            total_both += int(both_flags.sum())

    robust_rate = (
        category_robust / category_valid * 100
        if category_valid else 0
    )

    mahal_rate = (
        category_mahal / category_valid * 100
        if category_valid else 0
    )

    both_rate = (
        category_both / category_valid * 100
        if category_valid else 0
    )

    results.append({
        "attack_category": folder.name,
        "total_records": category_rows,
        "valid_records": category_valid,
        "invalid_records": category_invalid,
        "robust_detected": category_robust,
        "robust_detection_rate": robust_rate,
        "mahal_detected": category_mahal,
        "mahal_detection_rate": mahal_rate,
        "both_detected": category_both,
        "both_detection_rate": both_rate
    })

    print(
        f"{folder.name}: "
        f"records={category_rows:,}, "
        f"valid={category_valid:,}, "
        f"Robust={robust_rate:.2f}%, "
        f"Mahalanobis={mahal_rate:.2f}%"
    )

# Save category-wise results
output = pd.DataFrame(results)

output.to_csv(
    BASE / "attack_detection_results.csv",
    index=False
)

print("\n========== OVERALL ATTACK RESULTS ==========")
print("Total records:", f"{total_rows:,}")
print("Valid records:", f"{total_valid:,}")
print("Invalid records skipped:", f"{total_invalid:,}")

if total_valid:
    print(
        "Robust detection rate:",
        f"{total_robust / total_valid * 100:.4f}%"
    )

    print(
        "Mahalanobis detection rate:",
        f"{total_mahal / total_valid * 100:.4f}%"
    )

    print(
        "Both detectors detected:",
        f"{total_both / total_valid * 100:.4f}%"
    )

print("\nSaved: attack_detection_results.csv")
print("Attack evaluation completed.")
