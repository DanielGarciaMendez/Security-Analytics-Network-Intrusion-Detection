# Security Analytics – Network Intrusion Detection

**Project: Cross-Dataset Intrusion Detection**

## Project Goal
Can a machine-learning model trained to detect attacks in one network dataset also detect attacks in other network environments?

We are going to train a **Random Forest** model on **UNSW-NB15** and evaluate it on held-out UNSW-NB15 records, **CICIDS2017**, and **TON_IoT**. We will also compare **XGBoost** in the final project. Our main comparison changes the **test dataset**, and our primary metric is **AUROC**.


## Dataset Download and Reproducibility

This project uses three publicly available intrusion-detection datasets. 

To reproduce the experiments, download the following files from their official sources.

### 1. UNSW-NB15

**Source:** https://research.unsw.edu.au/projects/unsw-nb15-dataset

Download:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

Place both files in `data/raw/unsw_nb15/`.

### 2. CICIDS2017

**Source:** https://www.unb.ca/cic/datasets/ids-2017.html

Download `MachineLearningCSV.zip`, extract all eight CSV files, and place them in `data/raw/cicids2017/`.

### 3. TON_IoT

**Source:** https://research.unsw.edu.au/projects/toniot-datasets

Download the network traffic train/test CSV subset containing `train_test_network.csv`.

Place the file in `data/raw/ton_iot/`.

## Setup

Run these commands from the repository root in Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

The recorded run used Python 3.13.7. `requirements-lock.txt` records the exact
package versions; `requirements.txt` is the shorter, unpinned dependency list.

## Main workflow

After installing dependencies and placing the eleven raw CSVs in the paths above,
run this command from the repository root:

```powershell
.\.venv\Scripts\python.exe src\run_checkpoint.py
```

It runs the checkpoint audit, feature alignment, final splits, Random Forest,
course checker, and report generation, in that order. It writes `split.csv.gz`
and `scores.csv.gz` at the repository root. It rebuilds generated results and
the saved model; raw CSVs remain unchanged. XGBoost is not implemented yet.

The remaining commands below run individual stages or optional investigations.

## Where to find the current results

| Question | Main file |
|---|---|
| What data was loaded and what problems were found? | [Checkpoint audit](results/audits/checkpoint_2b/audit_table.md) |
| What rows were kept/excluded, and why? | [Final split report](results/splits/section_2c_final.md) |
| What are the official metrics and intervals? | [Course checker output](results/baseline/checker_output.txt) |
| What model, threshold, hardware and timing were used? | [Run metadata](results/baseline/run_metadata.json) |
| What goes into Sections 3–5 of the checkpoint? | [Results and finish plan](results/checkpoint/results_and_finish_plan.md) |
| What does the isolated README rebuild verify? | `results/reproduction/verification.json`, after the verification run finishes |

## Data Audits

The **primary dataset is UNSW-NB15** (the Random Forest training source). The audit (auditoverview2b.py) script reads the original UNSW-NB15 training and testing files, all eight CICIDS2017 CSVs, and the TON_IoT network CSV

Run the complete Section 2(b) audit:

```powershell
.\.venv\Scripts\python.exe src\auditoverview2b.py
```

**Main report:** [Section 2(b) — Dataset Audit Table](results/audits/checkpoint_2b/audit_table.md)

The audit script saves its reports in `results/audits/checkpoint_2b/`.

| File | Purpose |
|---|---|
| [`audit_table.md`](results/audits/checkpoint_2b/audit_table.md) | Checkpoint table with key-feature, duplicate and screening definitions. |
| `audit_by_file.csv` | Per-file dimensions, labels, and duplicate statistics. |
| `original_label_counts.csv` | Counts of the original labels, including individual CICIDS2017 attack types. |
| `key_feature_quality.csv` | Missing, nonnumeric, and infinite measurements in inspected features. |
| `audit_details.json` | Reproducible methods, detailed results, and limitations. |


### Detailed Per-File Audits 

Optional supporting exploration, separate from the main checkpoint pipeline:

```powershell
.\.venv\Scripts\python.exe src\audit_all.py
```

These supporting reports are saved in `results/audits/`.

## Split Counts (Sections 2(c) and 2(d))

After preparing the aligned features, create reproducible assignments:

```powershell
.\.venv\Scripts\python.exe src\prepare_shared_features.py
.\.venv\Scripts\python.exe src\create_checkpoint_splits.py
```

Read [the final counts and split policy](results/splits/section_2c_final.md).
`section_2c_preview.md` contains historical pre-split estimates; use the final report for actual counts.


```powershell
.\.venv\Scripts\python.exe check_checkpoint.py --split split.csv.gz --scores scores.csv.gz --threshold 0.9426192528735632
```

This is the threshold for the saved baseline. After retraining, use the new
validation-selected value in `results/baseline/checker_command.txt`.

## Random Forest baseline

Train on finalized assignments, select the lowest validation score threshold with
an empirical false-positive rate at most 5%, and score each retained test row:

```powershell
.\.venv\Scripts\python.exe src\train_checkpoint_baseline.py
```

Uses 200 trees, `min_samples_leaf=2`, seed 42, and all available CPU cores.
There is no scaling, imputation, or refitting on validation. Test data does not
select the model settings or threshold. The script writes `scores.csv.gz`,
`models/random_forest.joblib`, and `results/baseline/` with validation scores,
timing, versions, point metrics, a confusion matrix, and a preliminary report.

The course checker is a separate next step. Use the exact command saved in
`results/baseline/checker_command.txt`; the default training run does not execute
it. For a complete rerun including the checker, add `--run-checker`.
The checker's metrics and confidence intervals supply the checkpoint numbers.

## Checkpoint results and reproduction

The completed checker output is saved in `results/baseline/checker_output.txt`.
The checkpoint-ready Sections 3–5, including the dated finish plan, are generated
as `results/checkpoint/results_and_finish_plan.md` and a matching PDF. Keep the
completed dataset-paper citations from your existing checkpoint document.

Use the Setup and Main workflow commands above to reproduce the full run.

To regenerate only the results/plan after the checker finishes:

```powershell
.\.venv\Scripts\python.exe src\build_checkpoint_results.py
```

Default checker resampling is unchanged (1,000 UNSW and 200 per external test).
The saved threshold is `0.9426192528735632`. The report documents strict training
exclusions and row-bootstrap limitations. The proposed finish dates should be
reconciled with the final-project deadline and any additional assignment message.

## Repository structure and overlapping reports

| Location | Role |
|---|---|
| `src/` | Project scripts and regression tests. |
| `data/raw/` | Original downloads; excluded from Git. |
| `data/processed/aligned/` | Shared-feature data, before final exclusions. |
| `results/audits/checkpoint_2b/` | Main checkpoint audit and structured supporting evidence. |
| `results/audits/unsw_nb15/`, `cicids/`, `ton_iot/` | Optional detailed source-data investigations. |
| `results/splits/` | Final counts, exclusions, group membership, and leakage checks. |
| `results/baseline/` | Random Forest metadata, validation scores, checker output, and figure. |
| `results/checkpoint/` | Generated submission prose, tables, PDF and finish plan. |
| `results/reproduction/` | Isolated rebuild log and verification; scratch workspace is ignored by Git. |
| `models/` | Local fitted models; excluded from Git. |

Some overlap is intentional: CSV/JSON files support analysis and verification,
while Markdown/PDF files present the same evidence for readers. In particular:

- `results_and_finish_plan.pdf` and `.md` are two formats of the same document;
  `finish_plan.md` extracts Section 5.
- `baseline_report.md` summarizes the model; the checkpoint document expands
  that information into submission sections. `checker_output.txt` is the source
  for official metric numbers; `checker_metrics.json` is its parsed copy.
- `split_counts.csv` and `exclusion_counts.csv` supply the tables in
  `section_2c_final.md`; row-level gzip files provide traceability.
- `audit_overview.txt`, `audit_summary.csv`, `dataset_totals.txt`, and per-file
  audit reports repeat some original-data counts. Use the checkpoint audit as
  the main reference. Their duplicate definitions and scopes can differ.
- `section_2c_preview.md` and the overlap investigations are historical evidence,
  not final assignments. Keep `preview_checkpoint_splits.py`: current training
  and split scripts import its feature-validation helpers.

No historical reports are required to be deleted for the main workflow to run.
For a less cluttered future layout, shared helpers could move into their own
module, tests into `tests/`, and historical scripts/reports into an archive.
That reorganization has not been performed.

To test the README rebuild without replacing the current research outputs:

```powershell
.\.venv\Scripts\python.exe src\verify_readme_reproduction.py
```

This requires the current outputs as a comparison reference. It copies source
code into a new workspace, links the eleven raw input files, rebuilds all main
outputs and compares assignments, probabilities, predictions and printed checker
results. It checks the existing environment against the lock file; it does not
test package installation, a fresh Git clone, or dataset downloads. The retained
workspace must be moved before starting another isolated verification run.

Reproduction compares probabilities within `1e-12`, rather than gzip bytes
(gzip timestamps differ). The observed rerun had identical fitted tree arrays
and identical attack/benign predictions, but floating-point differences up to
`3.33e-16` in threaded probability sums. With tied scores, those differences can
slightly change rank metrics: CICIDS AUROC rounded to 0.5049 on the rerun versus
0.5048 in the saved baseline. Use the original saved scores and checker output
for the reported checkpoint numbers. Exact rank-score replay would require
deterministic probability aggregation; it has not been enabled in this baseline.

To repeat only the comparisons against a completed retained workspace:

```powershell
.\.venv\Scripts\python.exe src\verify_readme_reproduction.py --compare-only
```
