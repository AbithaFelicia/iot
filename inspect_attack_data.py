import pandas as pd
from pathlib import Path

attack_root = Path("data/raw/CSV")

print("========== ATTACK DATASET INSPECTION ==========")

total_files = 0

for folder in sorted(attack_root.iterdir()):

    if not folder.is_dir():
        continue

    if folder.name == "Benign_Final":
        continue

    csv_files = list(folder.glob("*.csv"))

    if csv_files:
        total_files += len(csv_files)

        print(
            f"{folder.name}: {len(csv_files)} CSV files"
        )

print("\nTotal attack CSV files:", total_files)

print("\n========== SAMPLE ATTACK FILES ==========")

shown = 0

for folder in sorted(attack_root.iterdir()):

    if not folder.is_dir() or folder.name == "Benign_Final":
        continue

    for file in sorted(folder.glob("*.csv")):

        df = pd.read_csv(file, nrows=5)

        print("\nFile:", file.name)
        print("Folder:", folder.name)
        print("Columns:", list(df.columns))
        print("Sample shape:", df.shape)

        if "label" in df.columns:
            print("Labels:", df["label"].unique())

        shown += 1

        if shown == 5:
            break

    if shown == 5:
        break

print("\nInspection completed.")
