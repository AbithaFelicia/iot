import pandas as pd
import joblib
import json
import time

from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

BASE = Path("data/supervised")
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

print("Loading training and testing data...")

train = pd.read_csv(BASE / "train.csv")
test = pd.read_csv(BASE / "test.csv")

X_train = train.drop(columns=["label", "attack_category"])
y_train = train["label"]

X_test = test.drop(columns=["label", "attack_category"])
y_test = test["label"]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Features:", X_train.shape[1])

print("\nTraining Decision Tree...")

model = DecisionTreeClassifier(
    max_depth=15,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

start = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start

print("Training completed.")

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

print("\n========== DECISION TREE RESULTS ==========")

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test, predictions,
    target_names=["Benign", "Attack"]
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

joblib.dump(model, RESULTS / "decision_tree_model.joblib")

report = classification_report(
    y_test, predictions,
    target_names=["Benign", "Attack"],
    output_dict=True
)

with open(RESULTS / "decision_tree_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

pd.DataFrame({
    "actual": y_test,
    "predicted": predictions,
    "attack_probability": probabilities,
    "attack_category": test["attack_category"]
}).to_csv(RESULTS / "decision_tree_predictions.csv", index=False)

print("\nModel and evaluation files saved in:", RESULTS)
