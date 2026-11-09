import pandas as pd
from pathlib import Path

folder = Path("data/supervised/results")

models = {
    "DT": "decision_tree_predictions.csv",
    "DT_No_HTTPS": "decision_tree_no_https_predictions.csv",
    "LR": "logistic_regression_predictions.csv",
    "LR_No_HTTPS": "logistic_regression_no_https_predictions.csv",
    "Hybrid": "hybrid_dt_lr_predictions.csv",
    "Hybrid_No_HTTPS": "hybrid_dt_lr_no_https_predictions.csv"
}

results = []

for name, filename in models.items():
    df = pd.read_csv(folder / filename)
    attacks = df[df["actual"] == 1]

    rates = attacks.groupby("attack_category").apply(
        lambda x: (x["predicted"] == 1).mean() * 100,
        include_groups=False
    )

    for category, rate in rates.items():
        results.append({
            "category": category,
            "model": name,
            "detection_rate": rate
        })

table = pd.DataFrame(results).pivot(
    index="category",
    columns="model",
    values="detection_rate"
)

table.to_csv(folder / "https_category_comparison.csv")

print("========== HTTPS ABLATION CATEGORY COMPARISON ==========")
print(table.round(2).to_string())

print("\nSaved https_category_comparison.csv")
