import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import os

os.makedirs("results", exist_ok=True)

# ---------- Load and align ----------
X = pd.read_csv("data/processed/expr_matrix.csv", index_col=0)
y = pd.read_csv("data/processed/stage_labels.csv", index_col=0).squeeze()

X = X[~X.index.duplicated(keep="first")]
common = X.index.intersection(y.index)
X = X.loc[common]
y = y.loc[common].map({"early": 0, "late": 1})

assert X.index.equals(y.index), "X and y are misaligned"

print(f"{X.shape[0]} patients, {X.shape[1]} genes")
print(f"early={int((y == 0).sum())}, late={int((y == 1).sum())}")

# ---------- log2(TPM+1) ----------
# Mann-Whitney is rank based so the transform does not change its p-values.
# It is applied here only so the fold change below is a log fold change.
Xlog = np.log2(X + 1)

early = Xlog[y == 0]
late = Xlog[y == 1]

# ---------- Per-gene Mann-Whitney U ----------
# Two sided, no normality assumption. Genes with zero variance cannot be
# tested and are assigned p = 1.
genes = Xlog.columns.to_numpy()
pvals = np.ones(len(genes))

early_vals = early.to_numpy()
late_vals = late.to_numpy()

for i in range(len(genes)):
    a = early_vals[:, i]
    b = late_vals[:, i]
    if a.std() == 0 and b.std() == 0:
        continue
    try:
        _, p = mannwhitneyu(a, b, alternative="two-sided")
        pvals[i] = p
    except ValueError:
        pvals[i] = 1.0

# ---------- Benjamini-Hochberg FDR ----------
# Testing ~20k genes at p<0.05 yields ~1000 false positives by chance,
# so raw p-values are not usable on their own.
reject, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

log2fc = late.mean(axis=0).to_numpy() - early.mean(axis=0).to_numpy()

de = pd.DataFrame({
    "gene": genes,
    "log2fc_late_minus_early": log2fc,
    "pval": pvals,
    "qval": qvals,
    "significant": reject,
}).set_index("gene")

n_sig = int(de["significant"].sum())
print(f"\nSignificant at FDR 5%: {n_sig} of {len(de)} genes ({n_sig/len(de):.1%})")

de_sorted = de.sort_values("qval")
print("\nTop 25 by adjusted p-value:")
print(de_sorted.head(25).to_string(float_format=lambda v: f"{v:.3g}"))

# ---------- Cross-check against the CV projection ----------
weights = pd.read_csv("results/gene_weights_cv.csv", index_col=0)
merged = de.join(weights[["mean_weight", "folds_in_top50", "abs_mean"]], how="inner")

print("\n" + "=" * 60)
print("Genes reaching the top 50 in all 5 folds:")
cols = ["log2fc_late_minus_early", "qval", "significant", "mean_weight"]
print(merged[merged["folds_in_top50"] == 5][cols].to_string(float_format=lambda v: f"{v:.3g}"))

print("\nGenes reaching the top 50 in 4 of 5 folds:")
print(merged[merged["folds_in_top50"] == 4][cols].to_string(float_format=lambda v: f"{v:.3g}"))

# Do the CV-stable genes hit significance more often than genes at large?
stable = merged["folds_in_top50"] >= 3
rate_stable = merged.loc[stable, "significant"].mean()
rate_rest = merged.loc[~stable, "significant"].mean()
print(f"\nSignificant among genes stable in >=3 folds: {rate_stable:.1%} (n={int(stable.sum())})")
print(f"Significant among all other genes:            {rate_rest:.1%}")



EFFECT = 0.5  # log2 fold change, roughly 1.4x
de["strong"] = de["significant"] & (de["log2fc_late_minus_early"].abs() >= EFFECT)

n_strong = int(de["strong"].sum())
print(f"\nFDR 5% AND |log2fc| >= {EFFECT}: {n_strong} genes")
print(de[de["strong"]].sort_values("qval").head(30).to_string(float_format=lambda v: f"{v:.3g}"))

# Direction of the surviving set
up = int((de.loc[de["strong"], "log2fc_late_minus_early"] > 0).sum())
print(f"  up in late: {up}, down in late: {n_strong - up}")

# ---------- Cancer/testis antigen program ----------
# The CV projection was dominated by CT antigen families. Test whether that
# set is enriched among significant genes, rather than checking genes singly.
CT_PREFIXES = ("MAGEA", "MAGEB", "MAGEC", "GAGE", "XAGE", "PAGE",
               "SPANX", "CSAG", "CTAG", "SSX", "CT45A", "CT47")
ct_mask = merged.index.to_series().str.startswith(CT_PREFIXES).to_numpy()

n_ct = int(ct_mask.sum())
ct_sig_rate = merged.loc[ct_mask, "significant"].mean()
other_sig_rate = merged.loc[~ct_mask, "significant"].mean()

print("\n" + "=" * 60)
print(f"Cancer/testis antigen genes found: {n_ct}")
print(f"  significant at FDR 5%: {ct_sig_rate:.1%}")
print(f"  background rate:       {other_sig_rate:.1%}")

print("\nCT antigen genes, sorted by adjusted p-value (top 20):")
ct = merged[ct_mask].sort_values("qval")
print(ct.head(20)[cols + ["folds_in_top50"]].to_string(float_format=lambda v: f"{v:.3g}"))

# ---------- Genes of prior interest ----------
print("\n" + "=" * 60)
print("Genes of prior interest:")
prior = ["MT1H", "MT1F", "MT1G", "FOXM1", "MYC", "E2F1", "CHRNA9"]
present = [g for g in prior if g in merged.index]
print(merged.loc[present, cols + ["folds_in_top50"]].to_string(float_format=lambda v: f"{v:.3g}"))

de_sorted.to_csv("results/differential_expression.csv")
print("\nSaved to results/differential_expression.csv")