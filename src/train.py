import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report

RANDOM_STATE = 42

# ---------- Load and align ----------
X = pd.read_csv("data/processed/expr_matrix.csv", index_col=0)
y = pd.read_csv("data/processed/stage_labels.csv", index_col=0).squeeze()

# Prep script already deduped and aligned; these are guards, not fixes.
X = X[~X.index.duplicated(keep="first")]
common = X.index.intersection(y.index)
X = X.loc[common]
y = y.loc[common].map({"early": 0, "late": 1})

assert X.index.equals(y.index), "X and y are misaligned"
assert not y.isna().any(), "unmapped stage labels present"

gene_names = X.columns.to_numpy()
X_values = X.to_numpy(dtype=float)
y_values = y.to_numpy()

print(f"{X_values.shape[0]} patients, {X_values.shape[1]} genes")
print(f"class balance: {np.bincount(y_values)} (early, late)")

# ---------- Pipeline ----------
# log2(TPM+1) is element-wise so it cannot leak, but keeping it in the pipeline
# means every model sees identical preprocessing.
log2p1 = FunctionTransformer(lambda a: np.log2(a + 1), feature_names_out="one-to-one")

def make_pipeline(clf):
    return Pipeline([
        ("log", log2p1),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=50, random_state=RANDOM_STATE)),
        ("clf", clf),
    ])

models = {
    "Logistic Regression": make_pipeline(LogisticRegression(max_iter=5000)),
    "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
    "XGBoost": make_pipeline(XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ---------- Baseline ----------
dummy = DummyClassifier(strategy="most_frequent")
dummy_acc = cross_val_score(dummy, X_values, y_values, cv=cv, scoring="accuracy")
print(f"\nMajority-class baseline accuracy: {dummy_acc.mean():.4f}")

# ---------- Evaluate ----------
print("\n" + "=" * 60)
for name, pipe in models.items():
    acc = cross_val_score(pipe, X_values, y_values, cv=cv, scoring="accuracy")
    auc = cross_val_score(pipe, X_values, y_values, cv=cv, scoring="roc_auc")
    preds = cross_val_predict(pipe, X_values, y_values, cv=cv)

    print(f"\n{name}")
    print(f"  Accuracy: {acc.mean():.4f} (+/- {acc.std():.4f})")
    print(f"  AUROC:    {auc.mean():.4f} (+/- {auc.std():.4f})")
    print(f"  Lift over baseline: {acc.mean() - dummy_acc.mean():+.4f}")
    print("  Confusion matrix (rows true, cols pred):")
    print(confusion_matrix(y_values, preds))
    print(classification_report(y_values, preds, target_names=["early", "late"], digits=3))

# ---------- Per-fold gene weight projection ----------
# Refit the LR pipeline on each training fold and project the 50 coefficients
# back through that fold's PCA loadings to get one weight per gene.
print("\n" + "=" * 60)
print("Per-fold gene weight projection\n")

fold_weights = []
for fold, (train_idx, _) in enumerate(cv.split(X_values, y_values), start=1):
    pipe = make_pipeline(LogisticRegression(max_iter=5000))
    pipe.fit(X_values[train_idx], y_values[train_idx])

    components = pipe.named_steps["pca"].components_   # (50, n_genes)
    coefs = pipe.named_steps["clf"].coef_[0]           # (50,)
    gene_weights = coefs @ components                  # (n_genes,)

    fold_weights.append(pd.Series(gene_weights, index=gene_names, name=f"fold{fold}"))

weights_df = pd.concat(fold_weights, axis=1)

# Stability: how many folds ranked each gene in its top 50 by absolute weight
TOP_N = 50
top_sets = [set(w.abs().nlargest(TOP_N).index) for w in fold_weights]
appearance = pd.Series(0, index=gene_names, dtype=int)
for s in top_sets:
    appearance[list(s)] += 1

summary = pd.DataFrame({
    "mean_weight": weights_df.mean(axis=1),
    "std_weight": weights_df.std(axis=1),
    "folds_in_top50": appearance,
})
summary["abs_mean"] = summary["mean_weight"].abs()

stable = summary[summary["folds_in_top50"] == 5].sort_values("abs_mean", ascending=False)

print(f"Genes in the top {TOP_N} in all 5 folds: {len(stable)}\n")
print(stable[["mean_weight", "std_weight", "folds_in_top50"]].head(25).to_string())

genes_of_interest = ["MT1H", "MT1F", "MT1G", "FOXM1", "MYC", "E2F1"]
present = [g for g in genes_of_interest if g in summary.index]
print("\nGenes of prior interest:")
print(summary.loc[present, ["mean_weight", "std_weight", "folds_in_top50"]].to_string())

summary.sort_values("abs_mean", ascending=False).to_csv("results/gene_weights_cv.csv")
print("\nSaved full weights to results/gene_weights_cv.csv")