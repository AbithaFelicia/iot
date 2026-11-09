
import pandas as pd
from pathlib import Path

base = Path("data/processed/splits")

# Attack category results
attacks = pd.read_csv(
    base / "attack_detection_results.csv"
)

# Reconstruct integer detection counts
valid = attacks["valid_records"]

robust_tp = (
    valid * attacks["robust_detection_rate"] / 100
).round().astype(int)

mahal_tp = (
    valid * attacks["mahal_detection_rate"] / 100
).round().astype(int)

both_tp = (
    valid * attacks["both_detection_rate"] / 100
).round().astype(int)

# OR: either detector flags the record
either_tp = robust_tp + mahal_tp - both_tp

# Benign validation scores
benign = pd.read_csv(
    base / "benign_validation.csv"
)

robust_threshold = float(
    (base / "robust_threshold_new.txt").read_text()
)

mahal_threshold = float(
    (base / "mahalanobis_threshold_new.txt").read_text()
)

robust_fp = (
    benign["top3_mean_z"] > robust_threshold
)

mahal_fp = (
    benign["mahalanobis_squared"] > mahal_threshold
)

both_fp = robust_fp & mahal_fp
either_fp = robust_fp | mahal_fp

total_attack = int(valid.sum())
total_benign = len(benign)

modes = {
    "Robust": (int(robust_tp.sum()), int(robust_fp.sum())),
    "Mahalanobis": (int(mahal_tp.sum()), int(mahal_fp.sum())),
    "AND": (int(both_tp.sum()), int(both_fp.sum())),
    "OR": (int(either_tp.sum()), int(either_fp.sum()))
}

print("========== DETECTOR MODE COMPARISON ==========")

for name, (tp, fp) in modes.items():

    fn = total_attack - tp
    tn = total_benign - fp

    recall = tp / total_attack * 100
    fpr = fp / total_benign * 100
    specificity = tn / total_benign * 100

    macro_recall = {
        "Robust": attacks["robust_detection_rate"].mean(),
        "Mahalanobis": attacks["mahal_detection_rate"].mean(),
        "AND": attacks["both_detection_rate"].mean(),
        "OR": (
            either_tp / valid * 100
        ).mean()
    }[name]

    print(f"\n{name}")
    print("TP:", tp)
    print("FN:", fn)
    print("FP:", fp)
    print("TN:", tn)
    print("Weighted attack detection rate:", round(recall, 4), "%")
    print("Macro attack detection rate:", round(macro_recall, 4), "%")
    print("Benign false-positive rate:", round(fpr, 4), "%")
    print("Benign specificity:", round(specificity, 4), "%")

print("\nComparison completed.")
