import pandas as pd
import os

# ---------- Load and clean clinical data ----------
clinical = pd.read_csv(
    "data/raw/clinical.project-tcga-kirc.2026-06-15/clinical.tsv",
    sep="\t"
)
clinical = clinical.replace("'--", pd.NA)

# ---------- Keep only primary disease diagnosis ----------
primary = clinical[clinical["diagnoses.diagnosis_is_primary_disease"] == True].copy()
assert len(primary) > 0, "primary diagnosis filter returned nothing, check the column dtype"
assert primary.groupby("cases.submitter_id")["diagnoses.ajcc_pathologic_stage"].nunique().max() <= 1

stage_labels = (
    primary
    .dropna(subset=["diagnoses.ajcc_pathologic_stage"])
    .drop_duplicates(subset=["cases.submitter_id"])
    .set_index("cases.submitter_id")["diagnoses.ajcc_pathologic_stage"]
)

# ---------- Map to binary labels ----------
stage_map = {
    "Stage I": "early", "Stage IA": "early", "Stage IB": "early", "Stage II": "early",
    "Stage III": "late", "Stage IIIA": "late", "Stage IIIB": "late", "Stage IV": "late",
}
y = stage_labels.map(stage_map)

unmapped = stage_labels[y.isna()].unique()
if len(unmapped) > 0:
    print(f"Warning: unmapped stage values dropped: {unmapped}")
y = y.dropna()

# ---------- Load sample sheet to map RNA files to patients ----------
samplesheet = pd.read_csv("data/raw/gdc_sample_sheet.2026-09-14.tsv", sep="\t")
sample = samplesheet[['File Name', 'Case ID', 'File ID']]

# ---------- Build gene expression matrix ----------
path = 'data/raw/rna_files'
expressions = []
missing = 0

for index, row in sample.iterrows():
    file_id = row['File ID']
    case_id = row['Case ID']
    file_path = os.path.join(path, file_id, row['File Name'])

    if not os.path.exists(file_path):
        missing += 1
        continue

    file_data = pd.read_csv(file_path, sep="\t", skiprows=1)
    file_data = file_data[file_data["gene_id"].str.contains('ENSG', na=False)]
    file_data = file_data[file_data["gene_type"] == "protein_coding"][['gene_name', 'tpm_unstranded']]

    # Gene symbols are not unique across Ensembl IDs. Sum TPM within a symbol
    # so set_index cannot produce a duplicated index.
    file_data = file_data.groupby('gene_name', as_index=False)['tpm_unstranded'].sum()

    series = file_data.set_index('gene_name')['tpm_unstranded']
    series.name = case_id
    expressions.append(series)

if missing:
    print(f"Warning: {missing} expression files listed in the sample sheet were not found on disk")

X = pd.concat(expressions, axis=1).T

# ---------- Collapse multiple aliquots per patient ----------
# Some TCGA cases were sequenced more than once. Average replicate profiles
# rather than arbitrarily keeping whichever the sample sheet listed first.
n_before = X.shape[0]
X = X.groupby(level=0).mean()
print(f"Collapsed {n_before} expression rows to {X.shape[0]} unique patients")

# ---------- Align to shared patients and save ----------
common = X.index.intersection(y.index)
X = X.loc[common]
y = y.loc[common]

assert X.index.equals(y.index), "X and y are misaligned after intersection"
assert not X.index.duplicated().any(), "duplicate patients remain in X"
assert not X.columns.duplicated().any(), "duplicate gene symbols remain in X"
assert X.isna().sum().sum() == 0, "NaNs present in expression matrix"

os.makedirs("data/processed", exist_ok=True)
X.to_csv("data/processed/expr_matrix.csv")
y.to_csv("data/processed/stage_labels.csv")

print(f"Saved: {X.shape[0]} patients, {X.shape[1]} genes")
print(f"Label counts:\n{y.value_counts()}")