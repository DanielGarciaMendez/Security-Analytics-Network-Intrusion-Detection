r"""COSC 4371 Project 11: consistent, reproducible, readable dataset audits.

Install this file as src/audit_all.py in your GitHub project, then run:
    .\.venv\Scripts\python.exe src\audit_all.py

Optional single-group runs:
    .\.venv\Scripts\python.exe src\audit_all.py --only unsw_train
    .\.venv\Scripts\python.exe src\audit_all.py --only unsw_test
    .\.venv\Scripts\python.exe src\audit_all.py --only cicids2017
    .\.venv\Scripts\python.exe src\audit_all.py --only ton_iot

The script only reads input CSVs and writes text/CSV reports. It does NOT clean,
modify, or delete any raw data. Exact duplicate counts are PER INPUT FILE,
NOT cross-file or train/test leakage counts. Those belong to a separate audit.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import textwrap

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "audits"

GROUPS = {
    "unsw_train": ("UNSW-NB15", "TRAINING FILE", "unsw_nb15"),
    "unsw_test": ("UNSW-NB15", "OFFICIAL TEST FILE", "unsw_nb15"),
    "cicids2017": ("CICIDS2017", "EXTERNAL TEST DATASET", "cicids"),
    "ton_iot": ("TON_IoT", "EXTERNAL TEST SUBSET", "ton_iot"),
}
WIDTH = 76


def sources_for(group: str) -> list[Path]:
    if group == "unsw_train":
        return [RAW / "unsw_nb15" / "UNSW_NB15_training-set.csv"]
    if group == "unsw_test":
        return [RAW / "unsw_nb15" / "UNSW_NB15_testing-set.csv"]
    if group == "cicids2017":
        return sorted((RAW / "cicids2017").rglob("*.csv"))
    if group == "ton_iot":
        return [RAW / "ton_iot" / "train_test_network.csv"]
    raise ValueError(group)


def fmt(n: int) -> str:
    return f"{n:,}"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def pct(n: int, total: int) -> str:
    return f"{100 * n / total:.2f}%" if total else "N/A"


def section(lines: list[str], number: int, title: str) -> None:
    lines.extend(["", f"{number}. {title.upper()}", "-" * WIDTH])


def label_breakdown(counts: Counter, total: int) -> list[str]:
    result = [f"  {'Original label':<36} {'Records':>13} {'Share':>9}",
              f"  {'-' * 36} {'-' * 13} {'-' * 9}"]
    for label, n in counts.most_common():
        result.append(f"  {label[:36]:<36} {fmt(n):>13} {pct(n, total):>9}")
    return result


def binary_summary(dataset: str, counts: Counter) -> tuple[int | None, int | None]:
    if dataset in {"UNSW-NB15", "TON_IoT"}:
        if not set(counts).issubset({"0", "1", "<MISSING>"}):
            return None, None
        return counts.get("0", 0), counts.get("1", 0)
    if dataset == "CICIDS2017":
        benign = sum(n for label, n in counts.items() if label.upper() == "BENIGN")
        attack = sum(n for label, n in counts.items() if label.upper() not in {"BENIGN", "<MISSING>"})
        return benign, attack
    return None, None


def analyze_file(path: Path, dataset: str) -> dict:
    # Work on one file at a time: never join 2.8m CICIDS rows in RAM.
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()
    if df.columns.duplicated().any():
        raise ValueError(f"Duplicate column names after trimming: {path}")

    labels = [c for c in df.columns if c.lower() == "label"]
    label_col = labels[0] if labels else None
    identifiers = [c for c in df.columns if c.lower().replace("_", " ") in {"id", "flow id"}]
    compare_cols = [c for c in df.columns if c not in identifiers]
    missing = {c: int(v) for c, v in df.isna().sum().items()}
    original_labels = Counter()
    if label_col:
        original_labels.update(df[label_col].astype("string").str.strip().fillna("<MISSING>").tolist())

    result = {
        "dataset": dataset,
        "path": path,
        "rows": len(df),
        "columns": list(df.columns),
        "label_column": label_col,
        "labels": original_labels,
        "missing_by_column": missing,
        "missing_target": missing.get(label_col, None),
        "missing_total": sum(missing.values()),
        "duplicate_full": int(df.duplicated().sum()),
        "duplicate_without_id": int(df.duplicated(subset=compare_cols).sum()),
        "excluded_identifiers": identifiers,
    }
    del df
    gc.collect()
    return result


def render_report(dataset: str, purpose: str, records: list[dict], combined: bool = False) -> str:
    total = sum(x["rows"] for x in records)
    counts = Counter()
    missing = Counter()
    for r in records:
        counts.update(r["labels"])
        missing.update(r["missing_by_column"])
    columns_all_match = all(r["columns"] == records[0]["columns"] for r in records)
    columns = records[0]["columns"]
    benign, attack = binary_summary(dataset, counts)
    total_miss = sum(r["missing_total"] for r in records)
    total_dup = sum(r["duplicate_without_id"] for r in records)
    total_dup_full = sum(r["duplicate_full"] for r in records)
    target_missing = sum(r["missing_target"] for r in records if r["missing_target"] is not None)

    lines = ["=" * WIDTH,
             f"DATA AUDIT  |  {dataset}  |  {purpose}",
             "=" * WIDTH,
             f"Files analyzed       : {len(records)}",
             f"Total records        : {fmt(total)}",
             f"Columns per file     : {len(columns) if columns_all_match else 'VARIES (inspect individual files)'}",
             f"Target column        : {records[0]['label_column'] or 'NOT FOUND'}",
             f"Missing target labels: {fmt(target_missing)}",
             "", "Source file(s):"]
    lines += [f"  - {rel(r['path'])}" for r in records]

    section(lines, 1, "Class distribution (original labels)")
    lines += label_breakdown(counts, total)
    if benign is not None:
        lines += ["", f"  Binary interpretation : benign = {fmt(benign)} ({pct(benign, total)})"
                  f"; attacks = {fmt(attack)} ({pct(attack, total)})"]
        if dataset == "CICIDS2017":
            lines += ["  Rule: BENIGN -> 0; every non-BENIGN attack name -> 1."
                      " Review any unexpected/missing labels before modeling."]
        else:
            lines += ["  Rule: label 0 -> benign; label 1 -> attack."]

    section(lines, 2, "Missing values")
    lines += [f"  Missing target labels : {fmt(target_missing)}",
              f"  Missing cells (all) : {fmt(total_miss)}"]
    nonzero = [(c, n) for c, n in missing.items() if n]
    if not nonzero:
        lines += ["  No standard null (NaN) values detected in inspected columns."]
    else:
        lines += ["  Columns containing missing values:"]
        lines += [f"    {c:<35} {fmt(n):>10}" for c, n in sorted(nonzero, key=lambda x: -x[1])]
        lines += [f"  {len(missing) - len(nonzero)} other column(s) have zero missing cells."]
    lines += ["  Note: Infinity and placeholder strings need separate quality checks."]

    section(lines, 3, "Exact duplicate records")
    lines += [f"  Identical full-row occurrences (within each file) : {fmt(total_dup_full)}",
              f"  Identical occurrences excluding ID fields      : {fmt(total_dup)}",
              f"  Fields excluded: {', '.join(records[0]['excluded_identifiers']) or 'none'}",
              "  IMPORTANT: These counts include target labels and are within-file ONLY.",
              "  They are NOT train/test leakage or cross-file duplicate counts.",
              "  Repeated summaries may represent distinct events; do not delete blindly."]
    if combined:
        lines += ["  The values above SUM each file's duplicate occurrences.",
                  "  Duplicates between different daily files are NOT included."]

    section(lines, 4, "Available columns")
    if columns_all_match:
        lines += [f"  {len(columns)} columns, in original file order:"]
        lines.extend("  " + chunk for chunk in textwrap.wrap(
            ", ".join(columns), width=WIDTH - 4, break_long_words=False,
            break_on_hyphens=False,
        ))
    else:
        lines += ["  Column schemas differ across files. See per-file reports."]

    if combined:
        section(lines, 5, "File-by-file summary")
        lines += [f"  {'#':>2}  {'Records':>11} {'Missing':>9} {'Dups*':>10}  CSV file",
                  f"  {'-' * 66}"]
        for i, r in enumerate(records, 1):
            lines.append(f"  {i:>2}  {fmt(r['rows']):>11} {fmt(r['missing_total']):>9} "
                         f"{fmt(r['duplicate_without_id']):>10}  {r['path'].name}")
        lines += ["  * Within-file duplicate occurrences (excluding ID fields)."]
        last_section = 6
    else:
        last_section = 5

    section(lines, last_section, "What this audit does NOT establish")
    lines += ["  - No train/validation/test overlap guarantee yet.",
              "  - No near-duplicate, feature-alignment or unit verification yet.",
              "  - No automatic cleaning or removal of rows has occurred.",
              "  - Check results/audits/quality_*.txt for separate quality checks",
              "    (or the corresponding files in dataset subfolders)."]
    return "\n".join(lines) + "\n"


def audit_group(group: str) -> list[dict]:
    dataset, purpose, folder = GROUPS[group]
    paths = sources_for(group)
    if not paths:
        raise FileNotFoundError(f"No CICIDS2017 CSV files found in {RAW / 'cicids2017'}")
    target_dir = OUT / folder
    target_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, path in enumerate(paths, start=1):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")
        print(f"Checking {path.name} ...", flush=True)
        r = analyze_file(path, dataset)
        rows.append(r)
        if group == "cicids2017":
            tag = f"cicids2017_{i:02d}_{path.stem}_audit.txt"
            (target_dir / tag).write_text(
                render_report(dataset, f"CSV {i} OF {len(paths)}", [r]), encoding="utf-8"
            )
        else:
            tag = f"{group}_audit.txt"
            (target_dir / tag).write_text(render_report(dataset, purpose, [r]), encoding="utf-8")

    if group == "cicids2017":
        overview = target_dir / "cicids2017_audit.txt"
        overview.write_text(render_report(dataset, purpose, rows, combined=True), encoding="utf-8")
        print(f"  Combined report: {rel(overview)}")
    elif group == "ton_iot":
        print(f"  Report: {rel(target_dir / 'ton_iot_audit.txt')}")
    else:
        print(f"  Report: {rel(target_dir / f'{group}_audit.txt')}")
    return rows


def write_overview(groups: dict[str, list[dict]]) -> None:
    all_rows = []
    text = ["=" * WIDTH, "PROJECT 11 - DATASET AUDIT OVERVIEW", "=" * WIDTH,
            "All rows below come from the downloaded CSV files on this machine.",
            "Duplicate counts are WITHIN each file, not cross-file leakage.", "",
            f"{'Subset':<22} {'Rows':>11} {'Attacks':>11} {'Benign':>11} {'Null cells':>11}",
            "-" * WIDTH]
    totals = {}
    for group, rows in groups.items():
        dataset, _purpose, _folder = GROUPS[group]
        n = sum(r['rows'] for r in rows)
        counts = Counter()
        for r in rows:
            counts.update(r['labels'])
        benign, attack = binary_summary(dataset, counts)
        text.append(f"{group:<22} {fmt(n):>11} {fmt(attack) if attack is not None else '?':>11} "
                    f"{fmt(benign) if benign is not None else '?':>11} "
                    f"{fmt(sum(r['missing_total'] for r in rows)):>11}")
        info = totals.setdefault(dataset, {"rows": 0, "files": 0, "labels": Counter()})
        info["rows"] += n
        info["files"] += len(rows)
        info["labels"].update(counts)
        for i, r in enumerate(rows, 1):
            all_rows.append({
                "group": group, "dataset": dataset, "file": rel(r['path']),
                "rows": r['rows'], "columns": len(r['columns']),
                "label_column": r['label_column'],
                "missing_target": r['missing_target'],
                "missing_all_cells": r['missing_total'],
                "duplicates_full_within_file": r['duplicate_full'],
                "duplicates_excluding_id_within_file": r['duplicate_without_id'],
                "excluded_identifier_columns": ";".join(r['excluded_identifiers']),
                "label_counts_json": json.dumps(dict(r['labels']), ensure_ascii=False),
            })
    text.extend(["", "Details:",
                 "  UNSW train : results/audits/unsw_nb15/unsw_train_audit.txt",
                 "  UNSW test  : results/audits/unsw_nb15/unsw_test_audit.txt",
                 "  CICIDS2017 : results/audits/cicids/cicids2017_audit.txt",
                 "  TON_IoT    : results/audits/ton_iot/ton_iot_audit.txt",
                 "", "See the separate quality_*.txt reports for cross-file overlap / inf / hosts."])
    (OUT / "audit_overview.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(OUT / "audit_summary.csv", index=False, encoding="utf-8")
    with (OUT / "dataset_totals.txt").open("w", encoding="utf-8") as f:
        for dataset, info in totals.items():
            f.write(f"{dataset}: {fmt(info['rows'])} rows across {info['files']} inspected file(s)\n")
            for label, n in info['labels'].most_common():
                f.write(f"  {label}: {fmt(n)} ({pct(n, info['rows'])})\n")
            f.write("  Reminder: counts of duplicates within CSV files are NOT cross-file overlap counts.\n\n")
    print("\nAll-dataset summary: results/audits/audit_overview.txt")



def format_existing_quality_reports() -> None:
    """Make existing QUALITY results readable without recomputing them.

    We keep quality_summary.json as the machine-readable source of truth.
    These are presentation-only reports; rerun check_quality_leakage.py to
    regenerate the underlying measurements when raw data changes.
    """
    source = OUT / "quality_summary.json"
    if not source.exists():
        print("No quality_summary.json found; quality report formatting skipped.")
        return
    data = json.loads(source.read_text(encoding="utf-8"))
    destinations = {
        "unsw": ("unsw_nb15", "quality_unsw.txt", "UNSW-NB15"),
        "cicids": ("cicids", "quality_cicids.txt", "CICIDS2017"),
        "ton": ("ton_iot", "quality_ton.txt", "TON_IoT"),
    }
    for group, (folder, name, dataset) in destinations.items():
        if group not in data:
            continue
        q = data[group]
        lines = ["=" * WIDTH,
                 f"DATA QUALITY CHECK  |  {dataset}",
                 "=" * WIDTH,
                 "Source: results/audits/quality_summary.json",
                 "These values were CALCULATED by check_quality_leakage.py.",
                 "This script reformats the saved results; it does not recompute them."]

        if group == "unsw":
            section(lines, 1, "Training vs official test: exact feature overlap")
            count = q.get("official_test_rows_matching_train_on_features", 0)
            test_n = q.get("test_rows", 0)
            lines += [
                f"  Original training rows  : {fmt(q.get('train_rows', 0))}",
                f"  Official testing rows   : {fmt(test_n)}",
                f"  Train rows matching test: {fmt(q.get('train_rows_matching_official_test_on_features', 0))}",
                f"  Test rows matching train: {fmt(count)} ({pct(count, test_n)} of test)",
                f"  Feature columns compared: {len(q.get('leakage_comparison_fields', []))}",
                "  The target label, attack category and row ID were NOT matching keys.",
                "  Concern: repeated feature patterns across training/testing partitions.",
                "  This is potential leakage; matching patterns may be distinct events.",
            ]
            section(lines, 2, "Duplicates within original files")
            lines += [
                f"  Train duplicates excluding ID (including label): {fmt(q.get('train_duplicate_occurrences_excluding_id_including_labels', 0))}",
                f"  Test duplicates excluding ID (including label) : {fmt(q.get('test_duplicate_occurrences_excluding_id_including_labels', 0))}",
                f"  Train feature-only duplicate occurrences       : {fmt(q.get('train_feature_duplicate_occurrences', 0))}",
                f"  Test feature-only duplicate occurrences        : {fmt(q.get('test_feature_duplicate_occurrences', 0))}",
            ]
            section(lines, 3, "Near-duplicate SCREEN (not proof)")
            near = q.get("near_duplicate_screen", {})
            if isinstance(near, dict):
                for field, value in near.items():
                    lines.extend([f"  {field}:"] + ["    " + line for line in textwrap.wrap(str(value), WIDTH-6)])
            section(lines, 4, "Invalid values in selected numeric features")
            for key, title in (("train_key_invalids", "TRAIN"), ("test_key_invalids", "TEST")):
                lines += [f"  {title}:"]
                for col, stat in q.get(key, {}).items():
                    lines.append(f"    {col:<12} missing={stat.get('missing', '?')}  "
                                 f"non_numeric={stat.get('non_numeric', '?')}  "
                                 f"infinite={stat.get('infinite', '?')}")
            extra_keys = ("entity_check", "time_check", "action")

        elif group == "cicids":
            section(lines, 1, "Cross-file fingerprint matches")
            lines += [
                f"  Cross-file duplicate CANDIDATE occurrences: {fmt(q.get('cross_file_duplicate_candidate_occurrences', 0))}",
                "  Candidates come from hashes of identical raw CSV rows.",
                "  Verify full rows if making cleaning decisions; not guaranteed by hashes.",
                "  Per-file candidate counts:",
            ]
            for rec in q.get("files", []):
                lines.append(f"    {rec.get('file', '?')}: "
                             f"{fmt(rec.get('matches_in_earlier_files_128bit_fingerprint', 0))}"
                             " matches in earlier file(s)")
            section(lines, 2, "Invalid or missing selected numeric measurements")
            lines += [f"  {'Feature':<34} {'Missing':>10} {'Nonnum':>10} {'Infinite':>10}"]
            for col, stat in q.get("selected_numeric_field_checks", {}).items():
                missing = stat.get("missing_or_blank", stat.get("missing", 0))
                invalid = stat.get("nonnumeric_nonblank", stat.get("non_numeric", 0))
                lines.append(f"  {col:<34} {fmt(missing):>10} {fmt(invalid):>10} "
                             f"{fmt(stat.get('infinite', 0)):>10}")
            extra_keys = ("cross_file_method", "entity_check", "time_check", "action")

        else:
            section(lines, 1, "Loaded file and duplicates")
            lines += [
                f"  File         : {q.get('file', 'unknown')}",
                f"  Rows         : {fmt(q.get('rows', 0))}",
                f"  Columns      : {fmt(q.get('columns', 0))}",
                f"  Exact full-row duplicate occurrences: {fmt(q.get('exact_duplicate_occurrences_all_columns', 0))}",
            ]
            section(lines, 2, "Repeated hosts and connections")
            for entity, stats in q.get("repeated_entity_counts", {}).items():
                lines += [f"  {entity}:",
                          f"    Unique values                : {fmt(stats.get('distinct_values', 0))}",
                          f"    Values appearing twice+      : {fmt(stats.get('values_repeated_at_least_twice', 0))}",
                          f"    Rows involving repeated value: {fmt(stats.get('rows_belonging_to_repeated_values', 0))}"]
            section(lines, 3, "Invalid values in selected numeric features")
            for col, stat in q.get("selected_numeric_field_checks", {}).items():
                lines.append(f"  {col:<16} missing={stat.get('missing', '?')}  "
                             f"non_numeric={stat.get('non_numeric', '?')}  "
                             f"infinite={stat.get('infinite', '?')}")
            extra_keys = ("time_check", "action")

        section(lines, 5 if group == "unsw" else (3 if group == "cicids" else 4), "Limitations and next actions")
        for k in extra_keys:
            v = q.get(k)
            if v:
                lines.append(f"  {k.replace('_', ' ').title()}:")
                lines.extend("    " + line for line in textwrap.wrap(str(v), width=WIDTH-6))
        lines += ["  Reminder: no raw data was edited by this formatting step."]
        target = OUT / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  Formatted quality report: {rel(target)}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate readable Project 11 dataset audit reports")
    parser.add_argument("--only", choices=["all", *GROUPS], default="all")
    args = parser.parse_args()
    selected = list(GROUPS) if args.only == "all" else [args.only]
    analyzed = {}
    for group in selected:
        analyzed[group] = audit_group(group)
    if args.only == "all":
        write_overview(analyzed)
    else:
        print("\nOnly one group audited: existing all-dataset summary files were NOT overwritten.")
    format_existing_quality_reports()
    print("Done. Raw CSVs were not modified.")


if __name__ == "__main__":
    main()
