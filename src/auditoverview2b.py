"""COSC 4371 Project 11 — Section 2(b): reproducible dataset audit table.

Run from repository root (Windows PowerShell):
    .\\.venv\\Scripts\\python.exe src\\build_checkpoint_audit.py

Inputs (read only):
    data/raw/unsw_nb15/UNSW_NB15_training-set.csv
    data/raw/unsw_nb15/UNSW_NB15_testing-set.csv
    data/raw/cicids2017/*.csv                  (eight official ML CSVs)
    data/raw/ton_iot/train_test_network.csv

Outputs under results/audits/checkpoint_2b/:
    audit_table.md             (copy into checkpoint)
    audit_table.html           (open in browser / print)
    audit_table.csv            (table cells as CSV)
    audit_by_file.csv          (one row per input CSV)
    original_label_counts.csv  (original labels, incl. CIC attack categories)
    key_feature_quality.csv   (missing, nonnumeric and infinite values)
    audit_details.json        (rules, exact fields, evidence and caveats)

Note: This script is an AUDIT, not a cleaning or train/validation/test split.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DEFAULT_OUT = ROOT / "results" / "audits" / "checkpoint_2b"
LABEL_COL = {"UNSW-NB15": "label", "CICIDS2017": "Label", "TON_IoT": "label"}
KEY_FIELDS = {
    "UNSW-NB15": ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate"],
    "CICIDS2017": ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Flow Bytes/s", "Flow Packets/s"],
    "TON_IoT": ["duration", "src_pkts", "dst_pkts", "src_bytes", "dst_bytes"],
}
MISSING = {"", "nan", "na", "n/a", "none", "null", "?"}
HASH_KEY_1 = "0123456789abcdef"
HASH_KEY_2 = "fedcba9876543210"


def sources() -> dict[str, list[Path]]:
    cic = sorted((RAW / "cicids2017").rglob("*.csv"))
    if len(cic) != 8:
        raise FileNotFoundError(
            f"Expected exactly eight CICIDS2017 MachineLearningCSV files in data/raw/cicids2017/, "
            f"but found {len(cic)}. Check for missing or extra CSVs before auditing."
        )
    return {
        "UNSW-NB15": [
            RAW / "unsw_nb15" / "UNSW_NB15_training-set.csv",
            RAW / "unsw_nb15" / "UNSW_NB15_testing-set.csv",
        ],
        "CICIDS2017": cic,
        "TON_IoT": [RAW / "ton_iot" / "train_test_network.csv"],
    }


def is_missing(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower().isin(MISSING)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalized_class(dataset: str, value: str) -> str:
    value = value.strip()
    if value.lower() in MISSING:
        return "missing"
    if dataset == "CICIDS2017":
        return "benign" if value.upper() == "BENIGN" else "attack"
    if value in {"0", "0.0"}:
        return "benign"
    if value in {"1", "1.0"}:
        return "attack"
    return "unknown"


def fingerprint_rows(frame: pd.DataFrame, fields: list[str]) -> list[int]:
    """Two independent pandas 64-bit fingerprints per raw-value record.

    Uses string-preserving CSV loads. The probability of 128-bit collision is
    negligible, but these are fingerprints, not byte-by-byte collision checks.
    """
    sub = frame.loc[:, fields]
    hash_a = pd.util.hash_pandas_object(sub, index=False, hash_key=HASH_KEY_1).to_numpy(dtype=np.uint64)
    hash_b = pd.util.hash_pandas_object(sub, index=False, hash_key=HASH_KEY_2).to_numpy(dtype=np.uint64)
    return [(int(a) << 64) | int(b) for a, b in zip(hash_a, hash_b)]


def dataset_audit(dataset: str, paths: list[Path], chunk_size: int):
    if not paths:
        raise FileNotFoundError(f"No CSVs found for {dataset}. Expected files in data/raw/.")
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"Missing {p}. Place your original CSVs as indicated in the README.")

    stats = {
        "dataset": dataset, "paths": [str(p.relative_to(ROOT)) for p in paths],
        "files": len(paths), "rows": 0, "columns": None,
        "raw_labels": Counter(), "classes": Counter(),
        "missing_target": 0,
        "missing_key": Counter(), "nonnumeric_key": Counter(), "infinite_key": Counter(),
        "duplicates_global": 0, "duplicates_within_file": 0,
        "rows_matching_earlier_files": 0, "duplicate_fields": [],
        "repeated_entities": {},
    }
    previous_file_hashes: set[int] = set()
    by_file: list[dict] = []
    expected_fields = None
    label_col = LABEL_COL[dataset]
    for path in paths:
        local_hashes: set[int] = set()
        file_rows = 0
        within = 0
        cross = 0
        file_missing_target = 0
        file_labels = Counter()
        print(f"Reading {dataset}: {path.name}", flush=True)
        for chunk in pd.read_csv(path, chunksize=chunk_size, dtype=str,
                                 keep_default_na=False, na_filter=False, low_memory=False,
                                 encoding="utf-8-sig"):
            chunk.columns = chunk.columns.str.strip()
            current_fields = chunk.columns.tolist()
            if len(set(current_fields)) != len(current_fields):
                raise ValueError(f"Duplicate column names in {path.name} after whitespace trimming.")
            if expected_fields is None:
                expected_fields = current_fields
                stats["columns"] = len(expected_fields)
                if label_col not in expected_fields:
                    raise ValueError(f"Missing target {label_col!r} in {path.name}")
                stats["duplicate_fields"] = [c for c in expected_fields if not (dataset == "UNSW-NB15" and c == "id")]
            elif current_fields != expected_fields:
                raise ValueError(f"Inconsistent schema/order in {path.name}. All files of {dataset} must share a schema.")

            n = len(chunk)
            file_rows += n
            stats["rows"] += n
            original = chunk[label_col].astype(str).str.strip()
            missing_target = int(is_missing(original).sum())
            stats["missing_target"] += missing_target
            file_missing_target += missing_target
            values = original.value_counts(dropna=False).to_dict()
            file_labels.update({str(k): int(v) for k, v in values.items()})
            stats["raw_labels"].update({str(k): int(v) for k, v in values.items()})
            for raw_label, count in values.items():
                stats["classes"][normalized_class(dataset, str(raw_label))] += int(count)

            for field in KEY_FIELDS[dataset]:
                if field not in chunk:
                    raise ValueError(f"Missing expected key feature {field!r} from {path.name}")
                value = chunk[field].astype(str).str.strip()
                missing_mask = is_missing(value)
                numeric = pd.to_numeric(value.mask(missing_mask), errors="coerce")
                stats["missing_key"][field] += int(missing_mask.sum())
                stats["nonnumeric_key"][field] += int((numeric.isna() & ~missing_mask).sum())
                stats["infinite_key"][field] += int(np.isinf(numeric.to_numpy(dtype="float64")).sum())

            for fp in fingerprint_rows(chunk, stats["duplicate_fields"]):
                if fp in local_hashes:
                    within += 1
                if fp in previous_file_hashes:
                    cross += 1
                if fp in local_hashes or fp in previous_file_hashes:
                    stats["duplicates_global"] += 1
                local_hashes.add(fp)

        stats["duplicates_within_file"] += within
        stats["rows_matching_earlier_files"] += cross
        previous_file_hashes.update(local_hashes)
        by_file.append({
            "dataset": dataset,
            "file": str(path.relative_to(ROOT)),
            "rows": file_rows,
            "columns": stats["columns"],
            "target": label_col,
            "missing_target": file_missing_target,
            "duplicate_occurrences_within_file": within,
            "rows_matching_previous_files": cross,
            "class_label_counts": json.dumps(dict(file_labels), ensure_ascii=False),
        })
        print(f"  {file_rows:,} rows; {within:,} within-file duplicate occurrences; "
              f"{cross:,} rows match earlier files", flush=True)

    return stats, by_file


def near_duplicate_screen(sample_n: int, seed: int) -> dict:
    """Reproduce the pre-cleaning UNSW training-only coarse-group screen.

    Counts *sample rows participating in candidate near-duplicate groups*, not
    pair counts and not confirmed copies of the same underlying flow.
    """
    path = RAW / "unsw_nb15" / "UNSW_NB15_training-set.csv"
    train = pd.read_csv(path, low_memory=False)
    train.columns = train.columns.str.strip()
    necessary = ["proto", "service", "state", "spkts", "dpkts", "dur", "sbytes", "dbytes"]
    if not all(f in train for f in necessary):
        missing = [f for f in necessary if f not in train]
        raise ValueError(f"Near-duplicate audit requires missing UNSW fields {missing}")
    sample = train.sample(n=min(len(train), sample_n), random_state=seed)
    groups = sample[["proto", "service", "state", "spkts", "dpkts"]].copy()
    groups["duration_rounded_0_01s"] = pd.to_numeric(sample["dur"], errors="coerce").round(2)
    total_bytes = pd.to_numeric(sample["sbytes"], errors="coerce") + pd.to_numeric(sample["dbytes"], errors="coerce")
    groups["total_bytes_nearest_100"] = (total_bytes / 100).round(0)
    feature_fields = [f for f in train if f not in {"id", "label", "attack_cat"}]
    groups["full_network_feature_fingerprint"] = pd.util.hash_pandas_object(sample[feature_fields], index=False).to_numpy()
    coarse_keys = [c for c in groups if c != "full_network_feature_fingerprint"]
    distinct_patterns = groups.groupby(coarse_keys, dropna=False)["full_network_feature_fingerprint"].transform("nunique")
    candidate_rows = int((distinct_patterns > 1).sum())
    return {
        "source_file": str(path.relative_to(ROOT)),
        "sample_size": int(len(sample)), "seed": int(seed),
        "candidate_rows": candidate_rows,
        "rule": "Same proto, service, state, spkts and dpkts; duration rounded to 0.01 seconds; total bytes rounded to nearest 100; at least two distinct full 42-feature network patterns within the group.",
        "interpretation": "Sample rows in candidate near-duplicate groups, NOT distinct pairs or confirmed duplicate events.",
    }


def nfmt(n: int) -> str:
    return f"{n:,}"


def count_pct(k: int, total: int) -> str:
    return f"{nfmt(k)} ({100 * k / total:.2f}%)" if total else "0 (n/a)"


def table_rows(s: dict, near: dict) -> list[tuple[str, str, str, str]]:
    u, c, t = s["UNSW-NB15"], s["CICIDS2017"], s["TON_IoT"]
    lines = [
        ("Rows", nfmt(u["rows"]), nfmt(c["rows"]), nfmt(t["rows"])),
        ("Columns", nfmt(u["columns"]), nfmt(c["columns"]), nfmt(t["columns"])),
        ("Benign count (%)", *(count_pct(x["classes"]["benign"], x["rows"]) for x in (u, c, t))),
        ("Malicious count (%)", *(count_pct(x["classes"]["attack"], x["rows"]) for x in (u, c, t))),
        ("Missing target values", *(nfmt(x["missing_target"]) for x in (u, c, t))),
        ("Missing key features", nfmt(sum(u["missing_key"].values())) + " [A]", "—", "—"),
        ("Exact duplicates", *(nfmt(x["duplicates_global"]) + " [B]" for x in (u, c, t))),
        ("Near-duplicates", f"{nfmt(near['candidate_rows'])} candidate rows / {nfmt(near['sample_size'])} sampled [C]", "—", "—"),
        ("Repeated entities", "Unavailable — no host IDs [D]", "—", "—"),
        ("Time range", "Unavailable — no timestamp [E]", "Not required [F]", "Not required [F]"),
    ]
    return lines


def output_reports(output: Path, stats: dict, by_file: list[dict], near: dict, chunk_size: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    matrix = table_rows(stats, near)
    flat = [{"Audit check": a, "UNSW-NB15": b, "CICIDS2017": c, "TON_IoT": d} for a, b, c, d in matrix]
    write_csv(output / "audit_table.csv", ["Audit check", "UNSW-NB15", "CICIDS2017", "TON_IoT"], flat)
    write_csv(output / "audit_by_file.csv", ["dataset", "file", "rows", "columns", "target", "missing_target", "duplicate_occurrences_within_file", "rows_matching_previous_files", "class_label_counts"], by_file)
    label_rows = []
    feature_rows = []
    for name, st in stats.items():
        for label, count in sorted(st["raw_labels"].items(), key=lambda z: (-z[1], z[0])):
            label_rows.append({"dataset": name, "original_label": label, "count": count, "percentage": round(100 * count / st["rows"], 4)})
        for field in KEY_FIELDS[name]:
            feature_rows.append({"dataset": name, "field": field, "missing": st["missing_key"][field], "nonnumeric_nonmissing": st["nonnumeric_key"][field], "infinite": st["infinite_key"][field]})
    write_csv(output / "original_label_counts.csv", ["dataset", "original_label", "count", "percentage"], label_rows)
    write_csv(output / "key_feature_quality.csv", ["dataset", "field", "missing", "nonnumeric_nonmissing", "infinite"], feature_rows)

    md = ["# Section 2(b) — Dataset Audit", "", "**Primary dataset:** UNSW-NB15. The baseline will train on its official training CSV.",
          "**Scope of this table:** All ORIGINAL loaded files were audited BEFORE cleaning: 2 UNSW CSVs, 8 CICIDS2017 CSVs and 1 TON_IoT network CSV. UNSW counts combine its official training and testing files, with file-level counts retained separately.",
          "", "| Audit check | UNSW-NB15 (primary) | CICIDS2017 | TON_IoT |",
          "|---|---:|---:|---:|"]
    for a, b, c, d in matrix:
        md.append(f"| {a} | {b} | {c} | {d} |")
    unsw_files = [f for f in by_file if f["dataset"] == "UNSW-NB15"]
    md.extend(["", "## UNSW-NB15 official partition detail", "",
               "The main audit table combines the two source files; the **baseline trains on the official training file only**.", "",
               "| Source file | Rows | Benign | Attack | Duplicate occurrences inside this file (excluding id) |",
               "|---|---:|---:|---:|---:|"])
    for f in unsw_files:
        labels = json.loads(f["class_label_counts"])
        md.append(f"| `{Path(f['file']).name}` | {nfmt(f['rows'])} | {nfmt(int(labels.get('0', 0)))} | {nfmt(int(labels.get('1', 0)))} | {nfmt(f['duplicate_occurrences_within_file'])} |")
    md.extend(["", "## Definitions and limitations", "",
               "**[A] Key features.** UNSW missing-value count is the sum across `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes` and `rate`. Missing counts were calculated on the loaded CSVs. For other datasets, the rubric does not require key-feature missingness; the separate `key_feature_quality.csv` nevertheless reports it.",
               "", "**[B] Exact duplicates.** Duplicate occurrences **after the first instance**, counted across all loaded CSVs *within each dataset*, using two independent 64-bit row fingerprints. Equality fields: UNSW = every loaded column **except `id`** (includes `attack_cat` and `label`); CICIDS2017 = **all 79 columns** (includes `Label`); TON_IoT = **all 44 columns** (includes `label` and `type`). Fingerprint collisions are theoretically possible. Counts may differ from older reports listing only *within-file* duplicates. These counts are NOT the train/test leakage check.",
               "", f"**[C] Near-duplicates (primary dataset only).** Sample of **{nfmt(near['sample_size'])} rows** from the UNSW official **training** file, random seed **{near['seed']}**. Rule: {near['rule']} Reported count is the number of **sample rows participating in candidate groups**, not the number of distinct pairs and not confirmation of the same event.",
               "", "**[D] Repeated entities.** Unavailable for the primary dataset: these prepared UNSW CSVs contain no stable source/destination host, device, or user identifier. The `id` column is only a row identifier; `ct_*` fields are aggregate statistics, not entity IDs.",
               "", "**[E] Time range.** Unavailable for UNSW because neither loaded, prepared CSV contains a per-record timestamp column. A publisher-stated collection period is not a computed time range from these rows.",
               "", "**[F] Other datasets.** A per-record time range is not required for CICIDS2017 and TON_IoT because this experiment uses a **dataset-boundary**, not time-boundary, split.",
               "", "## Additional evidence", "",
               "- `audit_by_file.csv`: original files, rows, columns, label counts, and duplicates inside each file.",
               "- `original_label_counts.csv`: **every original attack category** (important for CICIDS2017 before binary mapping).",
               "- `key_feature_quality.csv`: missing, nonnumeric, and infinite values in candidate network features for all three datasets.",
               "- `audit_details.json`: machine-readable methods, source filenames and results.",
               "", "**Status:** This is the pre-cleaning audit. Feature alignment, split assignments, row removals and final leakage checks are separate steps (Sections 2(c) and 2(d)).", ""])
    (output / "audit_table.md").write_text("\n".join(md), encoding="utf-8")

    trs = "\n".join("<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in row) + "</tr>" for row in matrix)
    footnotes = "\n".join(f"<p>{escape(line)}</p>" for line in md if line.startswith("**[") or line.startswith("**Status:"))
    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Section 2(b) — Dataset Audit</title>
<style>body{{font:14px/1.5 system-ui,Arial,sans-serif;color:#182536;margin:40px auto;max-width:1100px;padding:0 24px}}
h1{{font-size:25px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #d6dce5;padding:10px 12px;text-align:left;vertical-align:top}}th{{background:#edf3f8}}
tr:nth-child(even){{background:#fafbfc}}.note{{font-size:12px;color:#505d69}}@media print{{body{{margin:12mm;max-width:none}}table{{font-size:10px}}td,th{{padding:6px}}}}</style></head><body>
<h1>Section 2(b) — Dataset Audit</h1>
<p><b>Primary dataset:</b> UNSW-NB15 (Random Forest trains on the official training CSV).</p>
<p><b>Scope:</b> Original loaded data before cleaning. UNSW-NB15 includes official train and test files.</p>
<table><thead><tr><th>Audit check</th><th>UNSW-NB15</th><th>CICIDS2017</th><th>TON_IoT</th></tr></thead><tbody>{trs}</tbody></table>
<div class="note"><h2>Definitions and limitations</h2><p><b>[A]</b> UNSW key fields: dur, spkts, dpkts, sbytes, dbytes, rate.</p>
<p><b>[B]</b> Exact duplicates: repeated occurrences across each dataset's loaded files, comparing all columns except id for UNSW; all columns for CICIDS2017 and TON_IoT. Two independent 64-bit row fingerprints; collisions theoretically possible.</p>
<p><b>[C]</b> Candidate near-duplicates use a {near['sample_size']:,}-row UNSW training sample (seed {near['seed']}). Same protocol, service, state and directional packets; duration rounded to 0.01 s, total bytes to 100; at least two different full network patterns. Count is sample rows in candidate groups.</p>
<p><b>[D]</b> Repeated entities unavailable: no host/user/device identifier in prepared UNSW CSVs (id is a row number).</p>
<p><b>[E]</b> Primary dataset time range unavailable: no timestamp column in loaded UNSW CSV files.</p>
<p><b>[F]</b> External time ranges not required: dataset-boundary split, not time split.</p>
<p><b>Pre-cleaning audit only.</b> Splitting, cleaning and final leakage verification are not complete.</p></div></body></html>"""
    (output / "audit_table.html").write_text(html, encoding="utf-8")

    serializable = {}
    for name, st in stats.items():
        entry = dict(st)
        entry["raw_labels"] = dict(st["raw_labels"])
        entry["classes"] = dict(st["classes"])
        for field in ("missing_key", "nonnumeric_key", "infinite_key"):
            entry[field] = dict(st[field])
        entry["fingerprinting"] = "2 x 64-bit pandas hash_pandas_object over string-preserved CSV values, excluding row index"
        serializable[name] = entry
    (output / "audit_details.json").write_text(json.dumps({"random_seed": near["seed"], "read_chunk_size": chunk_size,
        "phase": "original_loaded_data_before_cleaning", "dataset_results": serializable,
        "near_duplicate_screen": near}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nAUDIT TABLE (generated from the raw files):")
    for a, b, c, d in matrix:
        print(f"{a:27} | {b:37} | {c:20} | {d}")
    print(f"\nReports written to: {output.relative_to(ROOT) if output.is_relative_to(ROOT) else output}")
    print("Open audit_table.html for a clean printable table or audit_table.md for Markdown.")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate COSC 4371 Section 2(b) audit table from raw CSVs")
    p.add_argument("--chunk-size", type=int, default=25000, help="Rows per chunk; reduce if RAM is limited")
    p.add_argument("--near-sample", type=int, default=20000, help="UNSW training-row subsample for near-duplicate screen")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.chunk_size < 100 or args.near_sample < 2:
        p.error("--chunk-size must be >=100 and --near-sample must be >=2")
    data = {}
    file_rows = []
    for name, paths in sources().items():
        stats, records = dataset_audit(name, paths, args.chunk_size)
        data[name] = stats
        file_rows.extend(records)
    screen = near_duplicate_screen(args.near_sample, args.seed)
    output_reports(DEFAULT_OUT, data, file_rows, screen, args.chunk_size)


if __name__ == "__main__":
    main()
