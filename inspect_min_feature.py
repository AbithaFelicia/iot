import pandas as pd
import numpy as np

base = "data/processed/splits/"

train = pd.read_csv(base + "train.csv")
cal = pd.read_csv(base + "calibration.csv")

for name, data in [("TRAINING", train), ("CALIBRATION", cal)]:

    print(f"\n========== {name}: MIN FEATURE ==========")

    print(data["Min"].describe(
        percentiles=[0.01, 0.05, 0.10, 0.25, 0.50,
                     0.75, 0.90, 0.95, 0.99, 0.999]
    ))

    print("\nMost frequent values:")

    print(data["Min"].value_counts().head(15))

    print("\nUnique values:", data["Min"].nunique())

    print("Zero values:", (data["Min"] == 0).sum())

    print("Values below 100:", (data["Min"] < 100).sum())

print("\nMin feature inspection completed.")
