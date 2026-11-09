import pandas as pd
import joblib
import json
import time

from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)

BASE = Path("data/supervised/file_based")
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

print("Loading datasets...")

train = pd.read_csv(BASE / "train.csv")
test = pd.read_csv(BASE / "test.csv")

metadata = ["label", "attack_category", "source_file"]

X_train = train.drop(columns=metadata + ["HTTPS"])
y_train = train["label"]

X_test = test.drop(columns=metadata + ["HTTPS"])
y_test = test["label"]

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and testing features do not match")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Features after removing HTTPS:", len(X_train.columns))

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

print("\nTraining Hybrid DT-LR without HTTPS...")

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

print("\n========== FILE-BASED HYBRID WITHOUT HTTPS ==========")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    target_names=["Benign", "Attack"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(
    model,
    RESULTS / "hybrid_dt_lr_file_based_no_https.joblib"
)

with open(
    RESULTS / "hybrid_dt_lr_file_based_no_https_metrics.json", "w"
) as f:
    json.dump(metrics, f, indent=4)

pd.DataFrame({
    "actual": y_test,
    "predicted": predictions,
    "attack_probability": probabilities,
    "attack_category": test["attack_category"],
    "source_file": test["source_file"]
}).to_csv(
    RESULTS / "hybrid_dt_lr_file_based_no_https_predictions.csv",
    index=False
)

print("\nSaved separate no-HTTPS experiment.")
