import pandas as pd
import numpy as np
from pathlib import Path

base = Path("data/processed/splits")

train = pd.read_csv(base / "train.csv")
cal = pd.read_csv(base / "calibration.csv")

features = ["Rate", "Min", "Variance"]

print("========== NON-NEGATIVE CHECK ==========")

for col in features:
    print(
        col,
        "| Training minimum:", train[col].min(),
        "| Calibration minimum:", cal[col].min()
    )

    if train[col].min() < 0 or cal[col].min() < 0:
        raise ValueError(f"{col} contains negative values.")

print("\n========== RAW VS LOG1P COMPARISON ==========")

for col in features:

    # Original scale
    raw_median = train[col].median()
    raw_mad = (train[col] - raw_median).abs().median()
    raw_scale = 1.4826 * raw_mad

    if raw_scale == 0:
        raw_scale = train[col].std()

    raw_z = (cal[col] - raw_median).abs() / raw_scale

    # Log-transformed scale
    train_log = np.log1p(train[col])
    cal_log = np.log1p(cal[col])

    log_median = train_log.median()
    log_mad = (train_log - log_median).abs().median()
    log_scale = 1.4826 * log_mad

    if log_scale == 0:
        log_scale = train_log.std()

    log_z = (cal_log - log_median).abs() / log_scale

    print(f"\nFeature: {col}")

    print("Original Z-score percentiles:")
    print(
        raw_z.quantile([0.50, 0.90, 0.95, 0.99, 0.999])
        .round(3).to_dict()
    )

    print("Log1p Z-score percentiles:")
    print(
        log_z.quantile([0.50, 0.90, 0.95, 0.99, 0.999])
        .round(3).to_dict()
    )

    print("Original maximum Z-score:", round(raw_z.max(), 3))
    print("Log1p maximum Z-score:", round(log_z.max(), 3))

print("\nComparison completed.")
