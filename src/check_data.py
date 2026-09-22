import pandas as pd

X = pd.read_csv("data/processed/expr_matrix.csv", index_col=0)
y = pd.read_csv("data/processed/stage_labels.csv", index_col=0).squeeze()

print("X shape:", X.shape)
print("y shape:", y.shape)
print("duplicate patients in X:", X.index.duplicated().sum())
print("duplicate genes in X columns:", X.columns.duplicated().sum())
print("label counts:\n", y.value_counts())
print("NaNs in X:", X.isna().sum().sum())