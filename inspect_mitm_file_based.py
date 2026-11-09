import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("data/supervised/file_based")

train = pd.read_csv(BASE / "train.csv")
test = pd.read_csv(BASE / "test.csv")
pred = pd.read_csv(
    BASE / "results/hybrid_dt_lr_file_based_predictions.csv"
)

if len(test) != len(pred):
    raise ValueError("Test and prediction row counts do not match")

if not (
    test["attack_category"].equals(pred["attack_category"])
    and test["source_file"].equals(pred["source_file"])
):
    raise ValueError("Test and prediction row order does not match")

test["predicted"] = pred["predicted"].values
test["attack_probability"] = pred["attack_probability"].values

mitm_train = train[
    train["attack_category"] == "MITM-ArpSpoofing"
]

mitm_test = test[
    (test["attack_category"] == "MITM-ArpSpoofing") &
    (test["label"] == 1)
]

features = [
    c for c in train.columns
    if c not in ["label", "attack_category", "source_file"]
]

print("========== MITM PREDICTION ANALYSIS ==========")
print("Training MITM records:", len(mitm_train))
print("Testing MITM records:", len(mitm_test))

print("\nCorrectly detected:", (mitm_test["predicted"] == 1).sum())
print("Missed attacks:", (mitm_test["predicted"] == 0).sum())

print("\nAttack probability summary:")
print(mitm_test["attack_probability"].describe())

print("\n========== FEATURE DISTRIBUTION SHIFT ==========")

rows = []

for feature in features:
    train_values = mitm_train[feature]
    test_values = mitm_test[feature]

    pooled_std = np.sqrt(
        (train_values.var() + test_values.var()) / 2
    )

    difference = (
        abs(train_values.mean() - test_values.mean()) / pooled_std
        if pooled_std > 0 and pd.notna(pooled_std)
        else 0
    )

    rows.append({
        "feature": feature,
        "train_mean": train_values.mean(),
        "test_mean": test_values.mean(),
        "standardized_difference": difference
    })

shift = pd.DataFrame(rows).sort_values(
    "standardized_difference",
    ascending=False
)

print(shift.head(10).to_string(index=False))

output = BASE / "results/mitm_feature_shift.csv"
shift.to_csv(output, index=False)

print("\nSaved feature analysis:", output)
