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
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

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
| [`audit_table.md`](results/audits/checkpoint_2b/audit_table.md) | Markdown table for the checkpoint, with definitions on how we considered key features, duplicat finding, etc |
| `audit_by_file.csv` | Per-file dimensions, labels, and duplicate statistics. |
| `original_label_counts.csv` | Counts of the original labels, including individual CICIDS2017 attack types. |
| `key_feature_quality.csv` | Missing, nonnumeric, and infinite measurements in inspected features. |
| `audit_details.json` | Reproducible methods, detailed results, and limitations. |


### Detailed Per-File Audits 

Ran this to understand data better, and overall more extensive look.

```powershell
.\.venv\Scripts\python.exe src\audit_all.py
```

These supporting reports are saved in `results/audits/`.