import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/supervised/binary_dataset_clean.csv")

X = df.drop(columns=["label", "attack_category"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

train = X_train.copy()
train["label"] = y_train
train["attack_category"] = df.loc[X_train.index, "attack_category"]

test = X_test.copy()
test["label"] = y_test
test["attack_category"] = df.loc[X_test.index, "attack_category"]

train.to_csv("data/supervised/train.csv", index=False)
test.to_csv("data/supervised/test.csv", index=False)

print("========== DATA SPLIT ==========")
print("Training records:", len(train))
print("Testing records:", len(test))

print("\nTraining class distribution:")
print(train["label"].value_counts().sort_index())

print("\nTesting class distribution:")
print(test["label"].value_counts().sort_index())

print("\nFeature count:", X.shape[1])
print("\nSaved training and testing datasets.")
