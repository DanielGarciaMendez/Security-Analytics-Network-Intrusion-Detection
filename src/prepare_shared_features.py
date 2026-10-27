"""Prepare conservative common network-flow features for Project 11.

Runs from anywhere; expects the project's data/raw directory as documented.
No raw CSVs are changed, no records are removed, and no model is trained.
Outputs feature-aligned CSV.gz files for later leakage-safe splitting.
"""
from __future__ import annotations

import gzip
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "aligned"
REPORT_DIR = ROOT / "results"
CHUNK_SIZE = 100_000

# All three sources contain bidirectional flow summaries.  We intentionally
# exclude 'bytes' for this first baseline because byte-accounting conventions
# can differ among Argus, CICFlowMeter, and Zeek.
SOURCES = [
    {
        "name": "UNSW training",
        "dataset": "UNSW-NB15",
        "files": [RAW / "unsw_nb15" / "UNSW_NB15_training-set.csv"],
        "output": "unsw_train.csv.gz",
        "columns": {"dur": "duration_seconds", "spkts": "forward_packets", "dpkts": "backward_packets"},
        "label": "label",
        "duration_divisor": 1,
        "prefix": "unsw_train",
    },
    {
        "name": "UNSW official testing",
        "dataset": "UNSW-NB15",
        "files": [RAW / "unsw_nb15" / "UNSW_NB15_testing-set.csv"],
        "output": "unsw_test.csv.gz",
        "columns": {"dur": "duration_seconds", "spkts": "forward_packets", "dpkts": "backward_packets"},
        "label": "label",
        "duration_divisor": 1,
        "prefix": "unsw_test",
    },
    {
        "name": "CICIDS2017 eight days",
        "dataset": "CICIDS2017",
        "files": sorted((RAW / "cicids2017").glob("*.csv")),
        "output": "cicids2017.csv.gz",
        "columns": {"Flow Duration": "duration_seconds", "Total Fwd Packets": "forward_packets", "Total Backward Packets": "backward_packets"},
        "label": "Label",
        "duration_divisor": 1_000_000,
        "prefix": "cicids2017",
    },
    {
        "name": "TON_IoT network subset",
        "dataset": "TON_IoT",
        "files": [RAW / "ton_iot" / "train_test_network.csv"],
        "output": "ton_iot.csv.gz",
        "columns": {"duration": "duration_seconds", "src_pkts": "forward_packets", "dst_pkts": "backward_packets"},
        "label": "label",
        "duration_divisor": 1,
        "prefix": "ton_iot",
    },
]


def transform_chunk(chunk: pd.DataFrame, config: dict, source: Path, row_start: int):
    chunk.columns = chunk.columns.str.strip()
    column_map = config["columns"]
    n = len(chunk)
    output = pd.DataFrame({
        "row_id": [f"{config['prefix']}:{source.stem}:{i}" for i in range(row_start, row_start + n)],
        "dataset": config["dataset"],
    })
    invalid = Counter()
    for origin, final in column_map.items():
        numbers = pd.to_numeric(chunk[origin], errors="coerce")
        inf_mask = np.isinf(numbers.to_numpy(dtype=float, na_value=np.nan))
        invalid[final] += int(numbers.isna().sum()) + int(inf_mask.sum())
        numbers = numbers.replace([np.inf, -np.inf], np.nan)
        if final == "duration_seconds":
            numbers = numbers / config["duration_divisor"]
        output[final] = numbers.to_numpy()

    raw_labels = chunk[config["label"]]
    if config["dataset"] == "CICIDS2017":
        cleaned = raw_labels.astype("string").str.strip().str.upper()
        if cleaned.isna().any() or cleaned.eq("").any():
            raise ValueError(f"Blank CICIDS2017 target label in {source.name}")
        output["label"] = cleaned.ne("BENIGN").astype(int).to_numpy()
    else:
        labels = pd.to_numeric(raw_labels, errors="raise")
        if labels.isna().any() or not labels.isin([0, 1]).all():
            raise ValueError(f"Unexpected target label in {source.name}")
        output["label"] = labels.astype(int).to_numpy()
    return output, invalid


def process(config: dict) -> str:
    files = config["files"]
    if not files or not all(file.is_file() for file in files):
        raise FileNotFoundError(f"Missing input CSVs for {config['name']}: {files}")
    expected = set(config["columns"]) | {config["label"]}
    destination = OUT / config["output"]
    rows = 0
    classes = Counter()
    invalids = Counter()
    header = True

    with gzip.open(destination, mode="wt", encoding="utf-8", newline="") as compressed_file:
        for file in files:
            header_names = {c.strip() for c in pd.read_csv(file, nrows=0).columns}
            missing = expected - header_names
            if missing:
                raise ValueError(f"{file.name} is missing expected columns: {sorted(missing)}")
            position = 0
            chunks = pd.read_csv(
                file,
                chunksize=CHUNK_SIZE,
                usecols=lambda col: col.strip() in expected,
                low_memory=False,
                encoding="latin-1",
            )
            for chunk in chunks:
                aligned, issues = transform_chunk(chunk, config, file, position)
                aligned.to_csv(compressed_file, header=header, index=False)
                header = False
                position += len(chunk)
                rows += len(chunk)
                classes.update(aligned["label"].value_counts().to_dict())
                invalids.update(issues)
            print(f"  Read {file.name}: {position:,} rows")

    report = (
        f"{config['name']}: {rows:,} records; "
        f"benign={classes[0]:,}; attack={classes[1]:,}; "
        f"invalid common feature cells={dict(invalids)}; "
        f"output={destination.relative_to(ROOT)}"
    )
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "SHARED FEATURE ALIGNMENT | PROJECT 11",
        "Features: duration_seconds, forward_packets, backward_packets",
        "CICIDS2017 microsecond duration converted to seconds (/ 1,000,000).",
        "No rows removed; blank/invalid feature values become NaN for later handling.",
        "No model training, split, imputation or duplicate cleaning performed here.",
        "Potential source-tool and packet accounting differences need documenting.",
        "",
    ]
    for config in SOURCES:
        print("Processing:", config["name"])
        lines.append(process(config))
    report = REPORT_DIR / "feature_alignment_summary.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("Saved:", report)


if __name__ == "__main__":
    main()
