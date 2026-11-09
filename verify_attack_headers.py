import pandas as pd
from pathlib import Path
import joblib

base = Path("data/processed/splits")
attack_root = Path("data/raw/CSV")

# Get the exact 35 feature names
scaler = joblib.load(
    base / "mahalanobis_scaler.joblib"
)

expected_features = list(scaler.feature_names_in_)

total_files = 0
valid_files = 0
invalid_files = []

label_columns = []

for folder in sorted(attack_root.iterdir()):

    if not folder.is_dir():
        continue

    if folder.name == "Benign_Final":
        continue

    for file in sorted(folder.glob("*.csv")):

        total_files += 1

        columns = pd.read_csv(
            file,
            nrows=0
        ).columns.tolist()

        missing = [
            col for col in expected_features
            if col not in columns
        ]

        if missing:
            invalid_files.append(
                (str(file), missing)
            )
        else:
            valid_files += 1

        if "label" in columns:
            label_columns.append(str(file))

print("========== ATTACK HEADER VERIFICATION ==========")

print("Total attack files:", total_files)
print("Valid files:", valid_files)
print("Invalid files:", len(invalid_files))

print("\nFiles containing a label column:", len(label_columns))

if invalid_files:
    print("\n========== INVALID FILES ==========")

    for filename, missing in invalid_files:
        print(filename)
        print("Missing features:", missing)

print("\nHeader verification completed.")
