import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("landmark_data.csv")
X = df.drop("label", axis=1)
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)
train = X_train.copy()
train["label"] = y_train
train["set"] = "train"
test = X_test.copy()
test["label"] = y_test
test["set"] = "test"
merged = pd.concat([train, test])
merged.to_csv("output_path0.1.csv", index=False)
