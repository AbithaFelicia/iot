import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("data/raw/CSV")
BENIGN_DIR = Path("data/processed/selected")
OUTPUT_DIR = Path("data/supervised")

PER_CATEGORY = 3000
CHUNK_SIZE = 50000
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

benign_files = sorted(BENIGN_DIR.glob("BenignTraffic*.pcap.csv"))

if not benign_files:
    raise FileNotFoundError("Selected benign CSV files not found")

features = pd.read_csv(benign_files[0], nrows=0).columns.tolist()

def update_sample(current, chunk, limit):
    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()
    combined = pd.concat([current, chunk], ignore_index=True)

    if len(combined) > limit:
        combined = combined.sample(
            n=limit, random_state=RANDOM_STATE
        )

    return combined.reset_index(drop=True)

# Sample benign traffic
print("Sampling benign traffic...")
benign_sample = pd.DataFrame(columns=features)

for file in benign_files:
    for chunk in pd.read_csv(
        file, usecols=features, chunksize=CHUNK_SIZE
    ):
        benign_sample = update_sample(
            benign_sample, chunk, PER_CATEGORY * 33
        )

print("Available benign sample:", len(benign_sample))

# Sample attack traffic category by category
attack_parts = []

for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir() or folder.name == "Benign_Final":
        continue

    print("Sampling:", folder.name)
    category_sample = pd.DataFrame(columns=features)

    for file in sorted(folder.glob("*.csv")):
        for chunk in pd.read_csv(
            file, usecols=features, chunksize=CHUNK_SIZE
        ):
            category_sample = update_sample(
                category_sample, chunk, PER_CATEGORY
            )

    category_sample["attack_category"] = folder.name
    category_sample["label"] = 1

    print("  Selected records:", len(category_sample))
    attack_parts.append(category_sample)

attack_df = pd.concat(attack_parts, ignore_index=True)

# Make benign count equal to attack count
target = len(attack_df)

if target > len(benign_sample):
    raise ValueError("Not enough benign records for balanced sampling")

benign_df = benign_sample.sample(
    n=target, random_state=RANDOM_STATE
).copy()

benign_df["attack_category"] = "BENIGN"
benign_df["label"] = 0

dataset = pd.concat(
    [benign_df, attack_df], ignore_index=True
)

dataset = dataset.sample(
    frac=1, random_state=RANDOM_STATE
).reset_index(drop=True)

output_file = OUTPUT_DIR / "binary_dataset.csv"
dataset.to_csv(output_file, index=False)

print("\n========== SAMPLING COMPLETE ==========")
print("Total records:", len(dataset))
print("\nClass distribution:")
print(dataset["label"].value_counts().sort_index())
print("\nAttack categories:", attack_df["attack_category"].nunique())
print("Feature count:", len(features))
print("Saved to:", output_file)
