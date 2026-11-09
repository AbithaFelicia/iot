import joblib
import pandas as pd
from pathlib import Path

folder = Path("data/supervised/results")

dt = joblib.load(folder / "decision_tree_model.joblib")
lr = joblib.load(folder / "logistic_regression_model.joblib")
hybrid = joblib.load(folder / "hybrid_dt_lr_model.joblib")

train = pd.read_csv("data/supervised/train.csv")
features = train.drop(
    columns=["label", "attack_category"]
).columns

results = pd.DataFrame({
    "feature": features,
    "Decision_Tree": dt.feature_importances_,
    "Logistic_Regression": abs(
        lr.named_steps["classifier"].coef_[0]
    ),
    "Hybrid_DT": hybrid.estimators_[0].feature_importances_,
    "Hybrid_LR": abs(
        hybrid.estimators_[1]
        .named_steps["classifier"].coef_[0]
    )
})

results.to_csv(
    folder / "feature_importance_comparison.csv",
    index=False
)

for column in results.columns[1:]:
    print(f"\n========== TOP 10: {column} ==========")
    print(
        results[["feature", column]]
        .sort_values(column, ascending=False)
        .head(10)
        .to_string(index=False)
    )

print("\nSaved feature importance comparison.")
