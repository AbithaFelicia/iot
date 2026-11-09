import pandas as pd
from pathlib import Path

base = Path("data/processed/splits")

# Load both detector scores
robust = pd.read_csv(base / "robust_score_comparison.csv")
mahal = pd.read_csv(base / "mahalanobis_scores.csv")

# Load calibrated thresholds
robust_threshold = 11.610371823861197

mahal_threshold = float(
    (base / "mahalanobis_threshold.txt").read_text()
)

# Generate predictions
robust_flag = robust["top3_mean_z"] > robust_threshold

mahal_flag = (
    mahal["mahalanobis_squared"] > mahal_threshold
)

# Compare predictions
comparison = pd.DataFrame({
    "robust_flag": robust_flag,
    "mahal_flag": mahal_flag
})

both_flagged = (robust_flag & mahal_flag).sum()
robust_only = (robust_flag & ~mahal_flag).sum()
mahal_only = (~robust_flag & mahal_flag).sum()
both_normal = (~robust_flag & ~mahal_flag).sum()

print("========== DETECTOR AGREEMENT ==========")

print("Total records:", len(comparison))
print("Both detectors flag:", both_flagged)
print("Only Robust Z-score flags:", robust_only)
print("Only Mahalanobis flags:", mahal_only)
print("Both classify as normal:", both_normal)

print("\n========== FLAGGING RATES ==========")

print(
    "Robust Z-score:",
    round(robust_flag.mean() * 100, 4), "%"
)

print(
    "Mahalanobis:",
    round(mahal_flag.mean() * 100, 4), "%"
)

print("\n========== AGREEMENT ==========")

agreement = (
    (robust_flag == mahal_flag).mean() * 100
)

print("Prediction agreement:", round(agreement, 4), "%")

print("\n========== SCORE CORRELATION ==========")

print(
    robust["top3_mean_z"].corr(
        mahal["mahalanobis_squared"]
    )
)

comparison.to_csv(
    base / "detector_agreement.csv",
    index=False
)

print("\nSaved: detector_agreement.csv")
print("Comparison completed.")
