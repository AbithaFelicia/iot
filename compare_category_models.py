import pandas as pd
from pathlib import Path

folder = Path("data/supervised/results")

models = {
    "Decision Tree": "decision_tree_predictions.csv",
    "Logistic Regression": "logistic_regression_predictions.csv",
    "Hybrid DT-LR": "hybrid_dt_lr_predictions.csv"
}

results = []

for model_name, filename in models.items():
    df = pd.read_csv(folder / filename)

    attacks = df[df["actual"] == 1].copy()

    category_result = (
        attacks.groupby("attack_category")
        .apply(
            lambda x: (x["predicted"] == 1).mean() * 100,
            include_groups=False
        )
    )

    for category, rate in category_result.items():
        results.append({
            "attack_category": category,
            "model": model_name,
            "detection_rate": rate
        })

comparison = pd.DataFrame(results)

table = comparison.pivot(
    index="attack_category",
    columns="model",
    values="detection_rate"
)

table = table.sort_values("Hybrid DT-LR")

print("========== CATEGORY-WISE MODEL COMPARISON ==========")
print(table.round(2).to_string())

table.to_csv(folder / "all_models_category_comparison.csv")

print("\nSaved all_models_category_comparison.csv")
