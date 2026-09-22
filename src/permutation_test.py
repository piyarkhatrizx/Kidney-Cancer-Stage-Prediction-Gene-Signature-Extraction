import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import os
import time

RANDOM_STATE = 42
N_PERMUTATIONS = 100

os.makedirs("results", exist_ok=True)

# ---------- Load and align ----------
X = pd.read_csv("data/processed/expr_matrix.csv", index_col=0)
y = pd.read_csv("data/processed/stage_labels.csv", index_col=0).squeeze()

X = X[~X.index.duplicated(keep="first")]
common = X.index.intersection(y.index)
X = X.loc[common]
y = y.loc[common].map({"early": 0, "late": 1})

assert X.index.equals(y.index), "X and y are misaligned"

X_values = X.to_numpy(dtype=float)
y_values = y.to_numpy()

print(f"{X_values.shape[0]} patients, {X_values.shape[1]} genes")
print(f"class balance: {np.bincount(y_values)} (early, late)")

# ---------- Same pipeline as train.py ----------
log2p1 = FunctionTransformer(lambda a: np.log2(a + 1), feature_names_out="one-to-one")

def make_pipeline():
    return Pipeline([
        ("log", log2p1),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=50, random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=5000)),
    ])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ---------- Observed AUROC ----------
observed = cross_val_score(
    make_pipeline(), X_values, y_values, cv=cv, scoring="roc_auc"
).mean()
print(f"\nObserved AUROC (real labels): {observed:.4f}")

# ---------- Null distribution ----------
# Shuffling the labels destroys any real relationship while leaving the data
# structure and the pipeline untouched. Whatever AUROC survives is what this
# pipeline manufactures from noise alone.
print(f"\nRunning {N_PERMUTATIONS} permutations (this will take a while)...")

rng = np.random.default_rng(RANDOM_STATE)
null_scores = np.empty(N_PERMUTATIONS)
start = time.time()

for i in range(N_PERMUTATIONS):
    y_shuffled = rng.permutation(y_values)
    null_scores[i] = cross_val_score(
        make_pipeline(), X_values, y_shuffled, cv=cv, scoring="roc_auc"
    ).mean()

    if (i + 1) % 10 == 0:
        elapsed = time.time() - start
        rate = elapsed / (i + 1)
        remaining = rate * (N_PERMUTATIONS - i - 1)
        print(f"  {i + 1}/{N_PERMUTATIONS} done "
              f"({elapsed/60:.1f} min elapsed, ~{remaining/60:.1f} min left)")

# ---------- Results ----------
# The permutation p-value uses the (k+1)/(n+1) form: a p-value can never be
# exactly zero, since the observed value is itself one draw from the null
# under the null hypothesis.
n_ge = int((null_scores >= observed).sum())
p_value = (n_ge + 1) / (N_PERMUTATIONS + 1)

print("\n" + "=" * 60)
print("Permutation test")
print(f"  Observed AUROC:     {observed:.4f}")
print(f"  Null mean:          {null_scores.mean():.4f}")
print(f"  Null std:           {null_scores.std():.4f}")
print(f"  Null min / max:     {null_scores.min():.4f} / {null_scores.max():.4f}")
print(f"  Null 95th pct:      {np.percentile(null_scores, 95):.4f}")
print(f"  Permutations >= observed: {n_ge} of {N_PERMUTATIONS}")
print(f"  p-value:            {p_value:.4f}")

z = (observed - null_scores.mean()) / null_scores.std()
print(f"  Observed is {z:.1f} SD above the null mean")

pd.DataFrame({
    "permutation": np.arange(1, N_PERMUTATIONS + 1),
    "auroc": null_scores,
}).to_csv("results/permutation_null.csv", index=False)

with open("results/permutation_summary.txt", "w") as f:
    f.write(f"observed_auroc\t{observed:.6f}\n")
    f.write(f"n_permutations\t{N_PERMUTATIONS}\n")
    f.write(f"null_mean\t{null_scores.mean():.6f}\n")
    f.write(f"null_std\t{null_scores.std():.6f}\n")
    f.write(f"null_p95\t{np.percentile(null_scores, 95):.6f}\n")
    f.write(f"n_ge_observed\t{n_ge}\n")
    f.write(f"p_value\t{p_value:.6f}\n")

print("\nSaved to results/permutation_null.csv and results/permutation_summary.txt")