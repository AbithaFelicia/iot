import pandas as pd
import numpy as np

BASE = "data/supervised/file_based/"

test = pd.read_csv(BASE + "test.csv")
pred = pd.read_csv(
    BASE + "results/hybrid_dt_lr_file_based_predictions.csv"
)

mitm = test[
    test["attack_category"] == "MITM-ArpSpoofing"
].copy()

mitm["predicted"] = pred.loc[mitm.index, "predicted"].values

features = [
    c for c in test.columns
    if c not in ["label", "attack_category", "source_file"]
]

detected = mitm[mitm["predicted"] == 1]
missed = mitm[mitm["predicted"] == 0]

print("Detected:", len(detected))
print("Missed:", len(missed))

rows = []

for feature in features:
    d_mean = detected[feature].mean()
    m_mean = missed[feature].mean()

    pooled_std = np.sqrt(
        (detected[feature].var() + missed[feature].var()) / 2
    )

    difference = (
        abs(d_mean - m_mean) / pooled_std
        if pooled_std > 0 and pd.notna(pooled_std)
        else 0
    )

    rows.append({
        "feature": feature,
        "detected_mean": d_mean,
        "missed_mean": m_mean,
        "standardized_difference": difference
    })

result = pd.DataFrame(rows).sort_values(
    "standardized_difference",
    ascending=False
)

print("\n========== FEATURES ASSOCIATED WITH MISSED ATTACKS ==========")
print(result.head(15).to_string(index=False))

result.to_csv(
    BASE + "results/mitm_missed_feature_analysis.csv",
    index=False
)

print("\nSaved analysis successfully.")
