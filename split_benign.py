import pandas as pd
from pathlib import Path

input_folder = Path("data/processed/selected")
output_folder = Path("data/processed/splits")

output_folder.mkdir(parents=True, exist_ok=True)

train_file = output_folder / "train.csv"
calibration_file = output_folder / "calibration.csv"

# Start fresh to avoid appending to old results
for file in [train_file, calibration_file]:
    if file.exists():
        file.unlink()

total_train = 0
total_calibration = 0

for file in sorted(input_folder.glob("*.csv")):

    print("\nSplitting:", file.name)

    df = pd.read_csv(file)

    # Randomly select 80% for training
    train = df.sample(frac=0.8, random_state=42)

    # Remaining 20% for calibration
    calibration = df.drop(train.index)

    # Shuffle calibration records
    calibration = calibration.sample(
        frac=1,
        random_state=42
    )

    # Save into combined CSV files
    train.to_csv(
        train_file,
        mode="a",
        index=False,
        header=not train_file.exists()
    )

    calibration.to_csv(
        calibration_file,
        mode="a",
        index=False,
        header=not calibration_file.exists()
    )

    total_train += len(train)
    total_calibration += len(calibration)

    print("Training records:", len(train))
    print("Calibration records:", len(calibration))

print("\n========== SPLIT SUMMARY ==========")
print("Training records:", total_train)
print("Calibration records:", total_calibration)
print("Total records:", total_train + total_calibration)
print("Training percentage:", round(total_train / (total_train + total_calibration) * 100, 2))
print("Calibration percentage:", round(total_calibration / (total_train + total_calibration) * 100, 2))

print("\nDataset splitting completed.")
