import pandas as pd
from pathlib import Path

input_folder = Path("data/processed/benign")
output_folder = Path("data/processed/selected")

output_folder.mkdir(parents=True, exist_ok=True)

remove_columns = [
    "Telnet",
    "SMTP",
    "Tot size",
    "LLC"
]

for file in sorted(input_folder.glob("*.csv")):

    print("\nProcessing:", file.name)

    df = pd.read_csv(file)

    df = df.drop(columns=remove_columns)

    output_file = output_folder / file.name

    df.to_csv(output_file, index=False)

    print("Records:", len(df))
    print("Selected features:", len(df.columns))

print("\nFeature selection completed.")
