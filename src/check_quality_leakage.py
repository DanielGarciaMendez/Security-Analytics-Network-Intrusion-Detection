"""COSC 4371 Project 11 — follow-up quality / overlap audit.

Run from the repository root (does NOT change raw CSVs):
  .\\.venv\\Scripts\\python.exe src\\check_quality_leakage.py --only unsw
  .\\.venv\\Scripts\\python.exe src\\check_quality_leakage.py --only all

Outputs: results/audits/quality_*.txt and quality_summary.json.
This checks the ORIGINAL dataset partitions, not the eventual model splits.
The later train/validation/test leakage check must be rerun after splitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "results" / "audits"
SEED = 42


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}. Check README paths.")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    if not df.columns.is_unique:
        raise ValueError(f"Duplicate column names after stripping whitespace: {path}")
    return df


def matched_rows(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> int:
    """Count rows in left exactly matching >=1 row in right on selected fields.

    The right side is deduplicated BEFORE merging to avoid multiplicative matches.
    Pandas treats missing keys as matching; we separately audit missing values.
    """
    keys = right[columns].drop_duplicates()
    result = left[columns].merge(keys.assign(_match=True), on=columns, how="left")
    return int(result["_match"].notna().sum())


def diagnostics(df: pd.DataFrame, numeric_fields: list[str]) -> dict:
    out = {}
    for col in numeric_fields:
        if col not in df:
            out[col] = {"status": "unavailable"}
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        invalid = int((s.isna() & df[col].notna()).sum())
        out[col] = {
            "missing": int(df[col].isna().sum()),
            "non_numeric": invalid,
            "infinite": int(np.isinf(s.to_numpy(dtype="float64")).sum()),
        }
    return out


def near_duplicate_candidates(df: pd.DataFrame, feature_cols: list[str], n: int) -> dict:
    """A clearly defined, approximate NEAR-DUPLICATE SCREEN, not a proof.

    A candidate pair must match proto/service/state/spkts/dpkts, have durations
    rounded to 0.01 s in the same bin and total bytes rounded to nearest 100
    in same bin, but differ on at least one full input feature.
    Only run on a reproducible random subsample if full scan is expensive.
    """
    must = ["proto", "service", "state", "spkts", "dpkts", "dur", "sbytes", "dbytes"]
    if not set(must).issubset(df.columns):
        return {"status": "unavailable", "reason": "Required comparison fields missing"}
    take = df.sample(n=min(n, len(df)), random_state=SEED).copy()
    coarse = take[["proto", "service", "state", "spkts", "dpkts"]].copy()
    coarse["duration_0.01_seconds"] = pd.to_numeric(take["dur"], errors="coerce").round(2)
    total_bytes = pd.to_numeric(take["sbytes"], errors="coerce") + pd.to_numeric(take["dbytes"], errors="coerce")
    coarse["total_bytes_nearest_100"] = (total_bytes / 100).round(0)
    # Fingerprint full model-input values (excluding labels and ID).
    # Hash collisions are possible in principle and should be verified before cleanup.
    full_hash = pd.util.hash_pandas_object(take[feature_cols], index=False)
    coarse["full_feature_fingerprint"] = full_hash.to_numpy()
    group_keys = [x for x in coarse.columns if x != "full_feature_fingerprint"]
    distinct_feature_patterns = coarse.groupby(group_keys, dropna=False)["full_feature_fingerprint"].transform("nunique")
    mask = distinct_feature_patterns > 1
    return {
        "sample_rows": int(len(take)),
        "rows_in_candidate_groups": int(mask.sum()),
        "different_feature_patterns_in_each_group_required": True,
        "rule": "Same proto, service, state, spkts, dpkts; duration rounded to 0.01 s; total bytes rounded to nearest 100; differing full input features",
        "note": "A screening heuristic. Some candidates may be unrelated flows. This does not establish cross-split near-duplicate leakage.",
    }


def unsw_audit(sample_size: int) -> dict:
    train = read_csv(RAW / "unsw_nb15" / "UNSW_NB15_training-set.csv")
    test = read_csv(RAW / "unsw_nb15" / "UNSW_NB15_testing-set.csv")
    feature_cols = [c for c in train if c not in {"id", "label", "attack_cat"}]
    if set(feature_cols) != set(c for c in test if c not in {"id", "label", "attack_cat"}):
        raise ValueError("UNSW train and test have different feature schemas")
    if not feature_cols:
        raise ValueError("No UNSW feature fields")
    feature_cols = sorted(feature_cols)

    overlap_train = matched_rows(train, test, feature_cols)
    overlap_test = matched_rows(test, train, feature_cols)
    # The published audit's definition kept label + attack_cat, excluding only ID.
    audit_cols = [c for c in train if c != "id"]
    out = {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "audit_duplicate_fields": audit_cols,
        "train_duplicate_occurrences_excluding_id_including_labels": int(train.duplicated(subset=audit_cols).sum()),
        "test_duplicate_occurrences_excluding_id_including_labels": int(test.duplicated(subset=audit_cols).sum()),
        "leakage_comparison_fields": feature_cols,
        "train_rows_matching_official_test_on_features": overlap_train,
        "official_test_rows_matching_train_on_features": overlap_test,
        "train_feature_duplicate_occurrences": int(train.duplicated(subset=feature_cols).sum()),
        "test_feature_duplicate_occurrences": int(test.duplicated(subset=feature_cols).sum()),
        "train_key_invalids": diagnostics(train, ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate"]),
        "test_key_invalids": diagnostics(test, ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate"]),
        "near_duplicate_screen": near_duplicate_candidates(train, feature_cols, sample_size),
        "entity_check": "Unavailable: the prepared UNSW CSVs do not contain source/destination host IDs; 'id' is a row ID, ct_* fields are aggregate counts, not stable host identities.",
        "time_check": "Unavailable: these prepared UNSW CSVs have no timestamp column. Document publisher collection period separately.",
        "action": "Do not train yet. If overlaps exist, preserve official test; exclude/move matching training rows, document counts, then split training by feature fingerprint groups and check both boundaries again.",
    }
    return out


def cic_audit(chunk_size: int) -> dict:
    files = sorted((RAW / "cicids2017").rglob("*.csv"))
    if not files:
        raise FileNotFoundError("No CICIDS2017 CSVs in data/raw/cicids2017/")
    # Stream chunks. Cross-FILE matches use 128 bits (two 64-bit hashes with
    # different keys) of raw string cell values, including label. These are
    # strong fingerprint candidates rather than mathematically verified exactness.
    numeric = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets",
               "Total Length of Fwd Packets", "Total Length of Bwd Packets",
               "Flow Bytes/s", "Flow Packets/s"]
    stats = {c: {"missing_or_blank": 0, "nonnumeric_nonblank": 0, "infinite": 0} for c in numeric}
    seen_prev_files: set[tuple[int, int]] = set()
    duplicate_from_earlier_files = 0
    file_summaries = []
    first_columns = None
    for path in files:
        this_file_hashes: set[tuple[int, int]] = set()
        matches = 0
        rows = 0
        for chunk in pd.read_csv(path, dtype=str, keep_default_na=False, chunksize=chunk_size):
            chunk.columns = chunk.columns.str.strip()
            if first_columns is None:
                first_columns = chunk.columns.tolist()
            elif first_columns != chunk.columns.tolist():
                raise ValueError(f"CICIDS2017 schema/order differs in {path.name}")
            rows += len(chunk)
            a = pd.util.hash_pandas_object(chunk, index=False, hash_key="0123456789abcdef").to_numpy()
            b = pd.util.hash_pandas_object(chunk, index=False, hash_key="fedcba9876543210").to_numpy()
            fingerprints = list(zip(map(int, a), map(int, b)))
            matches += sum(fp in seen_prev_files for fp in fingerprints)
            this_file_hashes.update(fingerprints)
            for col in numeric:
                if col not in chunk:
                    continue
                txt = chunk[col].str.strip()
                blank = txt.eq("") | txt.str.lower().isin(["nan", "na", "null", "none", "?"])
                parsed = pd.to_numeric(txt, errors="coerce")
                stats[col]["missing_or_blank"] += int(blank.sum())
                stats[col]["nonnumeric_nonblank"] += int((parsed.isna() & ~blank).sum())
                stats[col]["infinite"] += int(np.isinf(parsed.to_numpy()).sum())
        duplicate_from_earlier_files += matches
        seen_prev_files.update(this_file_hashes)
        file_summaries.append({"file": path.name, "rows": rows, "matches_in_earlier_files_128bit_fingerprint": matches})
    return {
        "files": file_summaries,
        "cross_file_duplicate_candidate_occurrences": duplicate_from_earlier_files,
        "cross_file_method": "Two 64-bit hashes of identical raw string-valued 79-column records, including Label. Cross-file only; within-file duplicates reported separately by audit_all.py. Fingerprint collisions theoretically possible.",
        "selected_numeric_field_checks": stats,
        "entity_check": "Unavailable from these ML CSV files: no source/destination IP or host ID columns.",
        "time_check": "Unavailable from these ML CSV files: no explicit per-row timestamp. Day filenames indicate scenario/day, not exact time range.",
        "action": "Document any invalid/inf values. Later align units, check shared-feature duplicates against UNSW train using the FINAL feature map, and choose/document a fixed test sample if necessary.",
    }


def repeat_stats(s: pd.Series) -> dict:
    sizes = s.fillna("<MISSING>").astype(str).value_counts(dropna=False)
    multi = sizes[sizes > 1]
    return {"distinct_values": int(len(sizes)), "values_repeated_at_least_twice": int(len(multi)),
            "rows_belonging_to_repeated_values": int(multi.sum())}


def ton_audit() -> dict:
    path = RAW / "ton_iot" / "train_test_network.csv"
    df = read_csv(path)
    fields = ["duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "src_port", "dst_port"]
    entities = {}
    for col in ["src_ip", "dst_ip"]:
        entities[col] = repeat_stats(df[col]) if col in df else {"status": "unavailable"}
    if {"src_ip", "dst_ip"}.issubset(df.columns):
        pair = df[["src_ip", "dst_ip"]].fillna("<MISSING>").astype(str).agg("|".join, axis=1)
        entities["source_destination_ip_pair"] = repeat_stats(pair)
    return {
        "file": str(path.relative_to(ROOT)),
        "rows": int(len(df)), "columns": int(len(df.columns)),
        "label_values": {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).items()},
        "selected_numeric_field_checks": diagnostics(df, fields),
        "repeated_entity_counts": entities,
        "exact_duplicate_occurrences_all_columns": int(df.duplicated().sum()),
        "time_check": "Unavailable: this downloaded CSV has no timestamp column; collection period belongs in dataset provenance.",
        "action": "Use this specific 211,043-row subset; do not describe it as the entire TON_IoT collection. If testing only, repeated host groups matter for interpreting independence, not for model-fitting splits on TON_IoT.",
    }


def save(name: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"quality_{name}.txt"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {path.relative_to(ROOT)}")
    if name == "unsw":
        print("  Training rows matching official testing rows on model features:", data["train_rows_matching_official_test_on_features"])
        print("  Official testing rows matching training rows on model features:", data["official_test_rows_matching_train_on_features"])
        print("  Near-duplicate screen:", data["near_duplicate_screen"])
    if name == "cicids":
        print("  Cross-file duplicate candidates:", data["cross_file_duplicate_candidate_occurrences"])
    if name == "ton":
        print("  Repeated source-IP records:", data["repeated_entity_counts"].get("src_ip"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["unsw", "cicids", "ton", "all"], default="unsw")
    p.add_argument("--near-sample", type=int, default=20_000)
    p.add_argument("--chunk-size", type=int, default=30_000)
    args = p.parse_args()
    results = {}
    which = ["unsw", "cicids", "ton"] if args.only == "all" else [args.only]
    for name in which:
        print(f"\n=== {name.upper()} FOLLOW-UP AUDIT ===", flush=True)
        if name == "unsw":
            data = unsw_audit(args.near_sample)
        elif name == "cicids":
            data = cic_audit(args.chunk_size)
        else:
            data = ton_audit()
        save(name, data)
        results[name] = data
    summary_path = OUT / "quality_summary.json"
    if args.only != "all" and summary_path.exists():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        existing.update(results)
        results = existing
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("NOTE: This does not generate final model splits or certify zero leakage. Recheck after cleaning/splitting and feature alignment.")


if __name__ == "__main__":
    main()
