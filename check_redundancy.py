import pandas as pd
from pathlib import Path
from itertools import combinations

folder = Path("data/processed/benign")

files = sorted(folder.glob("*.csv"))

first = pd.read_csv(files[0], nrows=1)
columns = first.columns.tolist()

candidate_pairs = set(combinations(columns, 2))

minimum = pd.Series(float("inf"), index=columns)
maximum = pd.Series(float("-inf"), index=columns)

for file in files:
    print("Checking:", file.name)

    for chunk in pd.read_csv(file, chunksize=50000):

        minimum = pd.concat(
            [minimum, chunk.min()], axis=1
        ).min(axis=1)

        maximum = pd.concat(
            [maximum, chunk.max()], axis=1
        ).max(axis=1)

        remaining = set()

        for a, b in candidate_pairs:
            if (chunk[a] == chunk[b]).all():
                remaining.add((a, b))

        candidate_pairs = remaining

print("\n========== CONSTANT FEATURES ==========")

for col in columns:
    if minimum[col] == maximum[col]:
        print(col)

print("\n========== EXACTLY IDENTICAL FEATURES ==========")

if candidate_pairs:
    for a, b in sorted(candidate_pairs):
        print(a, "<->", b)
else:
    print("No exactly identical feature pairs found.")

print("\nRedundancy check completed.")
