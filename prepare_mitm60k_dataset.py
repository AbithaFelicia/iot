import pandas as pd
from pathlib import Path

SOURCE = Path("data/supervised/file_based")
OUTPUT = Path("data/supervised/file_based_mitm60k")
MITM_LIMIT = 60000

MITM_FILE = Path(
    "data/raw/CSV/MITM-ArpSpoofing/"
    "MITM-ArpSpoofing.pcap.csv"
)

RANDOM_STATE = 42

print("Loading original training dataset...")

original = pd.read_csv(SOURCE / "train.csv")

metadata = ["label", "attack_category", "source_file"]
features = [c for c in original.columns if c not in metadata]

# Preserve every existing training record except the old MITM sample.
other_train = original[
    original["attack_category"] != "MITM-ArpSpoofing"
].copy()

print("Existing non-MITM training records:", len(other_train))

# Read the complete MITM training capture.
print("\nLoading MITM training capture...")

chunks = []

for chunk in pd.read_csv(
    MITM_FILE,
    usecols=features,
    chunksize=50000
):
    chunk = chunk.replace(
        [float("inf"), -float("inf")], float("nan")
    ).dropna()

    chunks.append(chunk)

mitm_data = pd.concat(chunks, ignore_index=True)

print("Valid MITM records available:", len(mitm_data))

if len(mitm_data) < MITM_LIMIT:
    raise ValueError("Not enough MITM records available")

# Uniform random sample from the complete source capture.
mitm_data = mitm_data.sample(
    n=MITM_LIMIT,
    random_state=RANDOM_STATE
).reset_index(drop=True)

mitm_data["source_file"] = MITM_FILE.name
mitm_data["attack_category"] = "MITM-ArpSpoofing"
mitm_data["label"] = 1

# Build the new training dataset.
new_train = pd.concat(
    [other_train, mitm_data],
    ignore_index=True
)

new_train = new_train.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# Preserve the original held-out test dataset exactly.
original_test = pd.read_csv(SOURCE / "test.csv")

new_train.to_csv(OUTPUT / "train.csv", index=False)
original_test.to_csv(OUTPUT / "test.csv", index=False)

print("\n========== MITM 30K DATASET ==========")
print("Training records:", len(new_train))
print("Testing records:", len(original_test))

print("\nTraining class distribution:")
print(new_train["label"].value_counts())

print("\nMITM training records:")
print(
    (new_train["attack_category"] == "MITM-ArpSpoofing").sum()
)

print("\nMITM testing records:")
print(
    (original_test["attack_category"] == "MITM-ArpSpoofing").sum()
)

print("\nSaved separate dataset:", OUTPUT)
