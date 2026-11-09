import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path("data/raw/CSV")
BENIGN_DIR = Path("data/processed/selected")
OUTPUT = Path("data/supervised/file_based")

PER_CATEGORY = 3000
CHUNK_SIZE = 50000
RANDOM_STATE = 42

OUTPUT.mkdir(parents=True, exist_ok=True)

benign_files = sorted(BENIGN_DIR.glob("BenignTraffic*.pcap.csv"))

features = pd.read_csv(benign_files[0], nrows=0).columns.tolist()


def sample_files(files, limit):
    current = pd.DataFrame()

    for file in files:
        for chunk in pd.read_csv(
            file,
            usecols=features,
            chunksize=CHUNK_SIZE
        ):
            chunk = chunk.replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            chunk["source_file"] = file.name

            current = pd.concat(
                [current, chunk], ignore_index=True
            )

            if len(current) > limit:
                current = current.sample(
                    n=limit,
                    random_state=RANDOM_STATE
                ).reset_index(drop=True)

    return current.reset_index(drop=True)


# Hold out one complete benign file
benign_test_file = benign_files[-1]
benign_train_files = benign_files[:-1]

print("Benign test file:", benign_test_file.name)

train_attack_parts = []
test_attack_parts = []
single_file_categories = []

for folder in sorted(ROOT.iterdir()):

    if not folder.is_dir() or folder.name == "Benign_Final":
        continue

    files = sorted(folder.glob("*.csv"))

    if len(files) >= 2:
        train_files = files[:-1]
        test_files = files[-1:]

        test_part = sample_files(test_files, PER_CATEGORY)

        test_part["attack_category"] = folder.name
        test_part["label"] = 1

        test_attack_parts.append(test_part)

        print(
            folder.name,
            "| Train files:", len(train_files),
            "| Test file:", test_files[0].name,
            "| Test records:", len(test_part)
        )

    else:
        train_files = files
        single_file_categories.append(folder.name)

        print(
            folder.name,
            "| Single source file: training only"
        )

    train_part = sample_files(train_files, PER_CATEGORY)

    train_part["attack_category"] = folder.name
    train_part["label"] = 1

    train_attack_parts.append(train_part)


train_attacks = pd.concat(
    train_attack_parts, ignore_index=True
)

test_attacks = pd.concat(
    test_attack_parts, ignore_index=True
)

print("\nTraining attack records:", len(train_attacks))
print("Testing attack records:", len(test_attacks))

# Balance benign and attack samples
train_benign = sample_files(
    benign_train_files, len(train_attacks)
)

test_benign = sample_files(
    [benign_test_file], len(test_attacks)
)

train_benign["attack_category"] = "BENIGN"
train_benign["label"] = 0

test_benign["attack_category"] = "BENIGN"
test_benign["label"] = 0

train_data = pd.concat(
    [train_benign, train_attacks],
    ignore_index=True
)

test_data = pd.concat(
    [test_benign, test_attacks],
    ignore_index=True
)

train_data = train_data.sample(
    frac=1, random_state=RANDOM_STATE
).reset_index(drop=True)

test_data = test_data.sample(
    frac=1, random_state=RANDOM_STATE
).reset_index(drop=True)

train_data.to_csv(OUTPUT / "train.csv", index=False)
test_data.to_csv(OUTPUT / "test.csv", index=False)

print("\n========== FILE-BASED DATASET COMPLETE ==========")
print("Training records:", len(train_data))
print("Testing records:", len(test_data))
print("Training source files:", train_data["source_file"].nunique())
print("Testing source files:", test_data["source_file"].nunique())
print("Single-file categories (not independently tested):")
print(single_file_categories)

print("\nSaved datasets in:", OUTPUT)
