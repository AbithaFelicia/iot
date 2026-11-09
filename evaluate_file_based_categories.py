import pandas as pd
from pathlib import Path

BASE = Path("data/supervised/file_based/results")

file = BASE / "hybrid_dt_lr_file_based_predictions.csv"
df = pd.read_csv(file)

attacks = df[df["actual"] == 1].copy()

attacks["detected"] = (
    attacks["actual"] == attacks["predicted"]
)

results = (
    attacks.groupby("attack_category")
    .agg(
        test_records=("actual", "count"),
        detected_attacks=("detected", "sum"),
        source_files=("source_file", "nunique")
    )
    .reset_index()
)

results["detection_rate"] = (
    results["detected_attacks"] /
    results["test_records"] * 100
)

results = results.sort_values("detection_rate")

output = BASE / "hybrid_file_based_category_results.csv"
results.to_csv(output, index=False)

print("========== FILE-BASED CATEGORY EVALUATION ==========")
print(results.to_string(index=False))

print("\nTotal tested attack categories:", len(results))
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

print("\nSaved to:", output)
