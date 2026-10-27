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

### Running the Data Audits

Install the required Python packages:

```powershell
python -m pip install -r requirements.txt
```

Execute the automated dataset audit:

```powershell
python src/audit_all.py
```

Audit results will be generated in `results/audits/`.


## Run the Audits
From the repository root, with the local virtual environment created and dependencies installed:

```powershell
.\.venv\Scripts\python.exe src\audit_all.py
```

The audit script saves its reports in `results/audits/`.

