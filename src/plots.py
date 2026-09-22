import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text

FIG_DIR = "results/figures"
import os
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- Plot 1: permutation null distribution ----------
null_df = pd.read_csv("results/permutation_null.csv")

summary = {}
with open("results/permutation_summary.txt") as f:
    for line in f:
        key, value = line.strip().split("\t")
        summary[key] = float(value)

observed = summary["observed_auroc"]
p_value = summary["p_value"]
z = (observed - summary["null_mean"]) / summary["null_std"]

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(null_df["auroc"], bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
ax.axvline(observed, color="#C44E52", linewidth=2.5)

# push the right edge out so there's room for the line and its label
xmax = max(null_df["auroc"].max(), observed)
ax.set_xlim(null_df["auroc"].min() - 0.02, xmax + 0.12)

ax.text(
    observed + 0.01, ax.get_ylim()[1] * 0.9,
    f"observed = {observed:.3f}\n{z:.1f} SD above null\np < 0.01" if p_value < 0.01
    else f"observed = {observed:.3f}\n{z:.1f} SD above null\np = {p_value:.3f}",
    color="#C44E52", fontsize=10, va="top", ha="left"
)
ax.set_xlabel("AUROC (shuffled labels)")
ax.set_ylabel("Count")
ax.set_title("Permutation test: observed AUROC vs null distribution")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/permutation_null.png", dpi=150)
plt.close(fig)

# ---------- Plot 2: volcano plot ----------
import numpy as np

de = pd.read_csv("results/differential_expression.csv", index_col=0)
de["neg_log10_q"] = -np.log10(de["qval"].clip(lower=1e-300))

strong = de["strong"] if "strong" in de.columns else (
    de["significant"] & (de["log2fc_late_minus_early"].abs() >= 0.5)
)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
    de.loc[~strong, "log2fc_late_minus_early"],
    de.loc[~strong, "neg_log10_q"],
    s=6, color="#B0B0B0", alpha=0.4, linewidths=0, label="not significant / small effect"
)
ax.scatter(
    de.loc[strong, "log2fc_late_minus_early"],
    de.loc[strong, "neg_log10_q"],
    s=10, color="#C44E52", alpha=0.7, linewidths=0, label="FDR 5% and |log2fc| >= 0.5"
)

ax.axvline(0.5, color="black", linewidth=0.8, linestyle="--")
ax.axvline(-0.5, color="black", linewidth=0.8, linestyle="--")
ax.axhline(-np.log10(0.05), color="black", linewidth=0.8, linestyle="--")

# ---------- Labeling with adjustText ----------


top_hits = de.loc[strong].sort_values("qval").head(10)

texts = [
    ax.annotate(
        gene, (row["log2fc_late_minus_early"], row["neg_log10_q"]),
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75)
    )
    for gene, row in top_hits.iterrows()
]

adjust_text(
    texts,
    arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
    expand=(1.3, 1.5),
)

# room on both axes so nothing sits on the border
xmax = de["log2fc_late_minus_early"].abs().max()
ax.set_xlim(-xmax * 1.15, xmax * 1.15)
ax.set_ylim(top=de["neg_log10_q"].replace([float("inf")], 0).max() * 1.15)
ax.set_xlabel("log2 fold change (late minus early)")
ax.set_ylabel("-log10(adjusted p value)")
ax.set_title("Differential expression: early vs late stage")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/volcano_plot.png", dpi=150)
plt.close(fig)

print(f"Saved to {FIG_DIR}/permutation_null.png and {FIG_DIR}/volcano_plot.png")