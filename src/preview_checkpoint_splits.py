"""COSC 4371 Project 11 — Section 2(c) *preview*, before final splitting.

From repository root, after running src/prepare_shared_features.py:
    .\\.venv\\Scripts\\python.exe src\\preview_checkpoint_splits.py

Reads aligned CSV.gz files and original UNSW source files. Does NOT edit data,
remove records, choose validation records or produce final split.csv.
Writes one readable evidence report: results/splits/section_2c_preview.md.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "data" / "processed" / "aligned"
RAW = ROOT / "data" / "raw" / "unsw_nb15"
OUT = ROOT / "results" / "splits" / "section_2c_preview.md"
FEATURES = ["duration_seconds", "forward_packets", "backward_packets"]
COLUMNS = ["row_id", "dataset", *FEATURES, "label"]
TESTS = [
    ("UNSW-NB15 official test", "unsw_test.csv.gz"),
    ("CICIDS2017", "cicids2017.csv.gz"),
    ("TON_IoT", "ton_iot.csv.gz"),
]
CHUNK = 100_000


def valid_feature_matrix(data: pd.DataFrame):
    """Canonical numeric input values without rounding/dropping zeroes."""
    cols = []
    for feat in FEATURES:
        v = pd.to_numeric(data[feat], errors="coerce").to_numpy(dtype="float64", copy=True)  # writable under pandas Copy-on-Write
        v[v == 0] = 0.0  # normalize negative zero for exact numeric equivalence
        cols.append(v)
    arr = np.column_stack(cols).astype("<f8", copy=False)
    valid = np.isfinite(arr).all(axis=1) & (arr >= 0).all(axis=1)
    return arr, valid


def keys_for(matrix: np.ndarray) -> list[bytes]:
    # Exact bytes for 3 numeric float64 feature values; avoids hash collisions.
    return [r.tobytes() for r in matrix]


def coarse_near_group(frame: pd.DataFrame) -> pd.Series:
    """Reuse Section 2(b) sampled near-duplicate *screening* bins for UNSW."""
    fields = ["proto", "service", "state", "spkts", "dpkts"]
    group = frame[fields].copy()
    group["dur_rounded_0.01s"] = pd.to_numeric(frame["dur"], errors="coerce").round(2)
    total = pd.to_numeric(frame["sbytes"], errors="coerce") + pd.to_numeric(frame["dbytes"], errors="coerce")
    group["total_bytes_nearest_100"] = (total / 100.0).round(0)
    # Small chance of 64-bit collision; sufficient as SCREEN only, not removal rule.
    return pd.util.hash_pandas_object(group, index=False)


def fmt(n):
    return f"{n:,}"


def pct(k, n):
    return f"{100 * k / n:.2f}%" if n else "n/a"


def main():
    train_path = ALIGNED / "unsw_train.csv.gz"
    expected = [train_path, *(ALIGNED / filename for _, filename in TESTS)]
    not_found = [p for p in expected if not p.is_file()]
    if not_found:
        missing = "\n".join(" - " + str(p.relative_to(ROOT)) for p in not_found)
        raise FileNotFoundError(
            "Missing aligned inputs:\n" + missing +
            "\nRun `.\\.venv\\Scripts\\python.exe src\\prepare_shared_features.py` first."
        )

    train = pd.read_csv(train_path, usecols=COLUMNS)
    if train["row_id"].duplicated().any():
        raise ValueError("Repeated row IDs in aligned UNSW training file")
    raw_train_path = RAW / "UNSW_NB15_training-set.csv"
    raw_test_path = RAW / "UNSW_NB15_testing-set.csv"
    if not raw_train_path.exists() or not raw_test_path.exists():
        raise FileNotFoundError("Missing UNSW original training/testing files in data/raw/unsw_nb15")
    rawfields = ["proto", "service", "state", "spkts", "dpkts", "dur", "sbytes", "dbytes"]
    raw_train = pd.read_csv(raw_train_path, usecols=rawfields, low_memory=False)
    raw_test = pd.read_csv(raw_test_path, usecols=rawfields, low_memory=False)
    if len(raw_train) != len(train):
        raise ValueError("Aligned train rows differ from UNSW original training rows. Check the source scripts.")

    train_arr, train_valid = valid_feature_matrix(train)
    train_keys = keys_for(train_arr[train_valid])
    train_set = set(train_keys)
    train_key_to_count = Counter(train_keys)
    train_key_to_label_counts = {}
    for key, label in zip(train_keys, train.loc[train_valid, "label"].astype(int)):
        if key not in train_key_to_label_counts:
            train_key_to_label_counts[key] = [0, 0]
        train_key_to_label_counts[key][label] += 1

    overall_overlap_keys = set()
    overlaps_by_test = []
    for label, filename in TESTS:
        path = ALIGNED / filename
        matches = set()
        test_rows = 0
        invalid_rows = 0
        matched_test_rows = 0
        class_counts = Counter()
        for chunk in pd.read_csv(path, usecols=COLUMNS, chunksize=CHUNK):
            test_rows += len(chunk)
            class_counts.update(chunk["label"].astype(int).value_counts().to_dict())
            arr, valid = valid_feature_matrix(chunk)
            invalid_rows += int((~valid).sum())
            matching_keys = [k for k in keys_for(arr[valid]) if k in train_set]
            matched_test_rows += len(matching_keys)
            matches.update(matching_keys)
        overlaps_by_test.append({
            "name": label, "rows": test_rows,
            "attacks": class_counts[1], "benign": class_counts[0],
            "invalid": invalid_rows,
            "matching_test_rows": matched_test_rows,
            "matching_train_rows": sum(train_key_to_count[k] for k in matches),
            "distinct_matching_patterns": len(matches),
        })
        overall_overlap_keys.update(matches)
        print(f"{label}: {fmt(test_rows)} test rows; {fmt(matched_test_rows)} match a train pattern", flush=True)

    matching_train_rows = sum(train_key_to_count[k] for k in overall_overlap_keys)
    invalid_training = int((~train_valid).sum())
    remaining_counts = Counter(train.loc[train_valid, "label"].astype(int).value_counts().to_dict())
    for key in overall_overlap_keys:
        zeroes, ones = train_key_to_label_counts[key]
        remaining_counts[0] -= zeroes
        remaining_counts[1] -= ones
    remaining_after_strict_exact = int(train_valid.sum()) - matching_train_rows

    raw_train_group = coarse_near_group(raw_train)
    raw_test_group = coarse_near_group(raw_test)
    seen_test_groups = set(raw_test_group.to_list())
    boundary_mask = raw_train_group.isin(seen_test_groups)
    near_coarse_test_train_rows = int(boundary_mask.sum())
    near_coarse_shared_groups = len(set(raw_train_group.to_list()) & seen_test_groups)

    md = [
        "# Section 2(c) — Pre-split boundary investigation",
        "",
        "**Status: PREVIEW ONLY.** No raw records were modified, no rows removed, and no final train/validation assignments made.",
        "The values below come from locally loaded, feature-aligned CSVs; they are not publisher statistics.",
        "",
        "## Shared feature mapping (candidate v0.1)",
        "",
        "| Canonical input | UNSW-NB15 | CICIDS2017 | TON_IoT | Unit conversion |",
        "|---|---|---|---|---|",
        "| `duration_seconds` | `dur` | `Flow Duration` | `duration` | CICIDS / 1,000,000; others unchanged |",
        "| `forward_packets` | `spkts` | `Total Fwd Packets` | `src_pkts` | None |",
        "| `backward_packets` | `dpkts` | `Total Backward Packets` | `dst_pkts` | None |",
        "",
        "The originator/forward packet definitions and flow timeouts may differ among collectors. Byte measures were withheld pending unit/definition verification. The candidate map must be approved before final splitting.",
        "",
        "## Source partitions and preliminary overlap impact",
        "",
        f"Original UNSW training source: **{fmt(len(train))} rows** "
        f"(benign={fmt(int((train['label']==0).sum()))}, attack={fmt(int((train['label']==1).sum()))}).",
        f"Invalid aligned UNSW training feature rows: **{fmt(invalid_training)}** (missing, infinite, or negative).",
        "",
        "| Planned test set | Original test rows | Benign | Attack | Invalid model inputs | Test rows matching a train pattern | Training rows matching this test set |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for o in overlaps_by_test:
        md.append(
            f"| {o['name']} | {fmt(o['rows'])} | {fmt(o['benign'])} | {fmt(o['attacks'])} | "
            f"{fmt(o['invalid'])} | {fmt(o['matching_test_rows'])} | {fmt(o['matching_train_rows'])} |"
        )
    md += [
        "",
        "**Definition:** A match requires exact numeric equality of *all three aligned model inputs* "
        "(without rounding or including labels). Numbers are counted as float64 after duration-unit conversion; "
        "negative zero is normalized to zero. Invalid model-input rows are not compared.",
        "Training-row counts for individual test sets **overlap**, so do not add them.",
        "",
        "## Hypothetical strict three-feature exclusion from UNSW training",
        "",
        f"Distinct train input patterns that occur in at least one test set: **{fmt(len(overall_overlap_keys))}**.",
        f"Training rows affected by >=1 test set: **{fmt(matching_train_rows)}** "
        f"({pct(matching_train_rows, len(train))} of the original training source).",
        f"Rows remaining *if* all matching training rows and invalid feature rows were excluded: "
        f"**{fmt(remaining_after_strict_exact)}** "
        f"(benign={fmt(remaining_counts[0])}, attack={fmt(remaining_counts[1])}).",
        "",
        "**This is not a deletion instruction.** With only three numerical features, unrelated flows can have "
        "the same measurements. Strict exclusion can remove representative behavior and bias the experiment. "
        "Consider the loss of each class before approving this rule.",
        "",
        "## Primary-dataset near-duplicate screening boundary",
        "",
        "UNSW coarse-group screening key: same `proto`, `service`, `state`, `spkts`, `dpkts`; "
        "duration rounded to 0.01 seconds, and total `sbytes + dbytes` rounded to nearest 100. "
        "This uses the original files and is deliberately only a candidate screen.",
        f"- Distinct UNSW coarse groups found in both official partitions: **{fmt(near_coarse_shared_groups)}**.",
        f"- Training records belonging to those common coarse groups: **{fmt(near_coarse_test_train_rows)}** "
        f"({pct(near_coarse_test_train_rows, len(raw_train))}).",
        "- Coarse-group overlap does not establish the same real event, and this count is not a recommended removal count.",
        "- This rule is wider than the audit's near-duplicate candidate rule because it does not require two different full feature vectors within each coarse group.",
        "",
        "## Before finalizing Section 2(c)",
        "",
        "1. Review the shared feature mapping and the potential strict-exclusion impact; document a defensible decision.",
        "2. Choose primary near-duplicate grouping/protection rule without treating every coarse match as the same event.",
        "3. Create group-aware UNSW training/validation assignments with a fixed seed and class counts.",
        "4. Preserve separate UNSW official, CICIDS2017 and TON_IoT test rows, or document any excluded test rows.",
        "5. Produce `split.csv`, count exclusions and which boundary they protect; then run Section 2(d) final leakage checks.",
        "",
        "**Do not call this output the final split table.** It is the measured evidence needed to choose the split rule.",
        "",
    ]
    if remaining_after_strict_exact < 0.5 * len(train):
        md.insert(md.index("## Primary-dataset near-duplicate screening boundary"),
                  "**CAUTION:** The strict three-feature rule would exclude over half of the original training records; "
                  "do not apply it blindly.\n")
    if remaining_counts[0] == 0 or remaining_counts[1] == 0:
        md.insert(md.index("## Primary-dataset near-duplicate screening boundary"),
                  "**CRITICAL:** Strict three-feature filtering would leave only one class, so a binary model cannot be trained with that rule.\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(md), encoding="utf-8")
    print("Saved", OUT.relative_to(ROOT))
    print("Strict three-feature filtering (NOT APPLIED) would leave", fmt(remaining_after_strict_exact), "train rows")
    print("Training rows sharing UNSW coarse near-screening groups with official test:", fmt(near_coarse_test_train_rows))


if __name__ == "__main__":
    main()
