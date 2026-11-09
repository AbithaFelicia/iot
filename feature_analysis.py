import pandas as pd
from pathlib import Path

folder = Path("data/processed/benign")

minimum = None
maximum = None
samples = []

for file in sorted(folder.glob("*.csv")):

    print("Analysing:", file.name)

    for chunk in pd.read_csv(file, chunksize=50000):

        chunk_min = chunk.min()
        chunk_max = chunk.max()

        if minimum is None:
            minimum = chunk_min
            maximum = chunk_max
        else:
            minimum = pd.concat([minimum, chunk_min], axis=1).min(axis=1)
            maximum = pd.concat([maximum, chunk_max], axis=1).max(axis=1)

        samples.append(chunk.sample(
            n=min(5000, len(chunk)),
            random_state=42
        ))

# Identify constant features
constant = minimum[minimum == maximum].index.tolist()

print("\n========== CONSTANT FEATURES ==========")
print(constant if constant else "No constant features found.")

# Combine samples for correlation analysis
sample_df = pd.concat(samples, ignore_index=True)

print("\nSample records:", len(sample_df))

print("\n========== FEATURE RANGES ==========")
print(pd.DataFrame({
    "Minimum": minimum,
    "Maximum": maximum
}).to_string())

print("\n========== HIGH CORRELATIONS ==========")

corr = sample_df.corr().abs()

for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        if corr.iloc[i, j] >= 0.95:
            print(
                corr.columns[i],
                "<->",
                corr.columns[j],
                ":",
                round(corr.iloc[i, j], 4)
            )

print("\nFeature analysis completed.")
