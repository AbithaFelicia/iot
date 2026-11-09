import pandas as pd
import joblib
import json
import time

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)

BASE = Path("data/supervised")
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

print("Loading datasets...")

train = pd.read_csv(BASE / "train.csv")
test = pd.read_csv(BASE / "test.csv")

X_train = train.drop(columns=["label", "attack_category"])
y_train = train["label"]

X_test = test.drop(columns=["label", "attack_category"])
y_test = test["label"]

decision_tree = DecisionTreeClassifier(
    max_depth=15,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

logistic_regression = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

model = VotingClassifier(
    estimators=[
        ("dt", decision_tree),
        ("lr", logistic_regression)
    ],
    voting="soft"
)

print("\nTraining Hybrid DT-LR...")

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
    "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
    "roc_auc": roc_auc_score(y_test, probabilities),
    "training_time_seconds": training_time
}

print("\n========== HYBRID DT-LR RESULTS ==========")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    target_names=["Benign", "Attack"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, RESULTS / "hybrid_dt_lr_model.joblib")

with open(RESULTS / "hybrid_dt_lr_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

pd.DataFrame({
    "actual": y_test,
    "predicted": predictions,
    "attack_probability": probabilities,
    "attack_category": test["attack_category"]
}).to_csv(
    RESULTS / "hybrid_dt_lr_predictions.csv",
    index=False
)

print("\nHybrid model and results saved in:", RESULTS)
