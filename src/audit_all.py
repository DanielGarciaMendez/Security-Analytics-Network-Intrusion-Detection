"""Project 11: preliminary reproducible audit of local CSVs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import gc
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "results" / "audits"

SOURCES = {
    "unsw_train": [("UNSW-NB15", RAW / "unsw_nb15" / "UNSW_NB15_training-set.csv")],
    "unsw_test": [("UNSW-NB15", RAW / "unsw_nb15" / "UNSW_NB15_testing-set.csv")],
    "cicids2017": [("CICIDS2017", p) for p in sorted((RAW / "cicids2017").rglob("*.csv"))],
    "ton_iot": [("TON_IoT", RAW / "ton_iot" / "train_test_network.csv")],
}


def audit(path: Path, dataset: str, tag: str) -> tuple[dict, Counter]:
    # Read one CSV at a time, rather than merging all CICIDS files in memory.
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate column names after trimming headers: {path}")

    labels = [c for c in df.columns if c.lower() == "label"]
    label_col = labels[0] if labels else None
    exclude_identifiers = [
        c for c in df.columns if c.lower().replace("_", " ") in {"id", "flow id"}
    ]
    comparison_cols = [c for c in df.columns if c not in exclude_identifiers]

    # These are exact pandas row comparisons within this file, not across files.
    full_dups = int(df.duplicated().sum())
    no_id_dups = int(df.duplicated(subset=comparison_cols).sum())
    missing = df.isna().sum()
    label_counter = Counter()
    if label_col:
        label_counter.update(df[label_col].astype("string").str.strip().fillna("<MISSING>").tolist())

    key_features = ["dur", "proto", "spkts", "dpkts", "sbytes", "dbytes"]
    key_missing = {c: int(missing[c]) for c in key_features if c in df.columns}

    lines = [
        f"DATASET: {dataset}",
        f"FILE: {path.relative_to(ROOT)}",
        f"ROWS: {len(df):,}",
        f"COLUMNS: {len(df.columns)}",
        f"COLUMN NAMES: {list(df.columns)}",
        f"TARGET COLUMN: {label_col if label_col else 'UNAVAILABLE / VERIFY MANUALLY'}",
        "ORIGINAL LABEL COUNTS (do not assume all unknown labels are benign):",
    ]
    for label, count in sorted(label_counter.items(), key=lambda kv: -kv[1]):
        pct = count / len(df) * 100 if len(df) else 0
        lines.append(f"  {label}: {count:,} ({pct:.2f}%)")
    lines += [
        f"MISSING TARGET VALUES: {int(missing[label_col]) if label_col else 'unavailable'}",
        f"MISSING VALUES (ALL COLUMNS): {int(missing.sum()):,}",
        "MISSING BY COLUMN:",
        *(f"  {c}: {int(n):,}" for c, n in missing.items()),
        f"MISSING KEY FEATURES (where present): {key_missing}",
        f"EXACT DUPLICATE ROWS INCLUDING IDENTIFIERS (within file): {full_dups:,}",
        f"IDENTIFIER FIELDS EXCLUDED FOR SECOND DUPLICATE CHECK: {exclude_identifiers}",
        f"DUPLICATE ROWS EXCLUDING IDENTIFIERS (within file): {no_id_dups:,}",
        "NOTE: Repeated flow summaries can represent distinct events. Do not delete blindly.",
        "NOTE: Cross-file duplicates, near-duplicates, entity repeats, timestamps and leakage require further checks.",
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = OUTPUT / f"{tag}_audit.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    row = {
        "group": tag,
        "dataset": dataset,
        "file": str(path.relative_to(ROOT)),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "label_column": label_col or "UNAVAILABLE",
        "missing_target": int(missing[label_col]) if label_col else "",
        "missing_all_cells": int(missing.sum()),
        "duplicates_full_within_file": full_dups,
        "duplicates_excluding_id_within_file": no_id_dups,
        "excluded_identifier_columns": ";".join(exclude_identifiers),
        "label_counts_json": json.dumps(dict(label_counter), ensure_ascii=False),
    }
    print(f"{path.name}: {len(df):,} rows, {len(df.columns)} columns; "
          f"{no_id_dups:,} duplicate occurrences without ID -> {report.relative_to(ROOT)}", flush=True)
    del df
    gc.collect()
    return row, label_counter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Project 11 CSV audits; results saved automatically")
    parser.add_argument("--only", choices=[*SOURCES, "all"], default="all")
    args = parser.parse_args()
    groups = list(SOURCES) if args.only == "all" else [args.only]
    output_rows = []
    totals = {}
    missing_files = []
    for group in groups:
        entries = SOURCES[group]
        if not entries:
            missing_files.append(f"No CSVs found under data/raw/{'cicids2017' if group == 'cicids2017' else group}")
            continue
        for i, (dataset, path) in enumerate(entries, start=1):
            if not path.is_file():
                missing_files.append(str(path.relative_to(ROOT)))
                continue
            tag = (group if len(entries) == 1 else f"{group}_{i:02d}_{path.stem}")
            row, labels = audit(path, dataset, tag)
            output_rows.append(row)
            info = totals.setdefault(dataset, {"rows": 0, "files": 0, "labels": Counter()})
            info["rows"] += row["rows"]
            info["files"] += 1
            info["labels"].update(labels)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    if output_rows:
        pd.DataFrame(output_rows).to_csv(OUTPUT / "audit_summary.csv", index=False)
        with (OUTPUT / "dataset_totals.txt").open("w", encoding="utf-8") as out:
            for dataset, info in totals.items():
                out.write(f"{dataset}: {info['rows']:,} total rows across {info['files']} inspected file(s)\n")
                for label, n in info["labels"].most_common():
                    out.write(f"  {label}: {n:,} ({100*n/info['rows']:.2f}%)\n")
                out.write("  Warning: Within-file duplicate counts cannot be used as cross-file duplicate counts.\n")
        print(f"\nSaved reports and tables in: {OUTPUT}")
    if missing_files:
        print("\nMISSING FILES (confirm names / folder structure):")
        for p in missing_files:
            print(" -", p)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
