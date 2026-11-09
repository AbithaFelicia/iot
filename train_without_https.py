import pandas as pd
import joblib
import json
import time

from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, roc_auc_score,
    confusion_matrix
)

BASE = Path("data/supervised")
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

print("Loading datasets...")

train = pd.read_csv(BASE / "train.csv")
test = pd.read_csv(BASE / "test.csv")

X_train = train.drop(columns=["label", "attack_category", "HTTPS"])
y_train = train["label"]

X_test = test.drop(columns=["label", "attack_category", "HTTPS"])
y_test = test["label"]

print("Features used:", len(X_train.columns))
print("HTTPS removed successfully.")

dt = DecisionTreeClassifier(
    max_depth=15,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

lr = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

hybrid = VotingClassifier(
    estimators=[("dt", dt), ("lr", lr)],
    voting="soft"
)

models = {
    "decision_tree_no_https": dt,
    "logistic_regression_no_https": lr,
    "hybrid_dt_lr_no_https": hybrid
}

all_metrics = {}

for name, model in models.items():

    print(f"\n========== TRAINING {name} ==========")

    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_test, predictions
        ),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "training_time_seconds": training_time
    }

    all_metrics[name] = metrics

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(model, RESULTS / f"{name}_model.joblib")

    pd.DataFrame({
        "actual": y_test,
        "predicted": predictions,
        "attack_probability": probabilities,
        "attack_category": test["attack_category"]
    }).to_csv(
        RESULTS / f"{name}_predictions.csv",
        index=False
    )

with open(RESULTS / "no_https_metrics.json", "w") as f:
    json.dump(all_metrics, f, indent=4)

print("\nAll three no-HTTPS experiments completed.")
print("Results saved in:", RESULTS)
