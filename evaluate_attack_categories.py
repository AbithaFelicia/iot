import pandas as pd
from pathlib import Path

file = Path(
    "data/supervised/results/hybrid_dt_lr_predictions.csv"
)

df = pd.read_csv(file)

attacks = df[df["actual"] == 1].copy()

attacks["detected"] = (
    attacks["actual"] == attacks["predicted"]
)

results = (
    attacks.groupby("attack_category")
    .agg(
        test_records=("actual", "count"),
        detected_attacks=("detected", "sum")
    )
    .reset_index()
)

results["detection_rate"] = (
    results["detected_attacks"] /
    results["test_records"] * 100
)

results = results.sort_values("detection_rate")

results.to_csv(
    "data/supervised/results/hybrid_attack_category_results.csv",
    index=False
)

print("========== ATTACK CATEGORY EVALUATION ==========")
print(results.to_string(index=False))

print("\nTotal categories:", len(results))
print(
    "Average category detection rate:",
    round(results["detection_rate"].mean(), 2), "%"
)
print(
    "Lowest detection rate:",
    round(results["detection_rate"].min(), 2), "%"
)
print(
    "Highest detection rate:",
    round(results["detection_rate"].max(), 2), "%"
)

print("\nSaved category evaluation results.")
