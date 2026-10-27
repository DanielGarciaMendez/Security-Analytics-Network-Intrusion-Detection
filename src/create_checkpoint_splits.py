"""Create checkpoint assignments without modifying raw or aligned data.

Run from the repository root:
    .venv/Scripts/python.exe src/create_checkpoint_splits.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from preview_checkpoint_splits import FEATURES, TESTS, valid_feature_matrix, keys_for

ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "data/processed/aligned"
OUT = ROOT / "results/splits"
SEED = 42
CHUNK = 100_000
META = ["row_id", "dataset", "label"]


def near_keys(raw):
    """Exact tuple keys, not hashes; labels and IDs never define a group."""
    fields = [c for c in raw if c not in {"id", "label", "attack_cat", "dur", "sbytes", "dbytes"}]
    if len(fields) != 39:
        raise ValueError("Expected 39 unchanged UNSW network features")
    group = raw[fields].copy()
    group["duration_bin"] = raw["dur"].round(2)
    group["total_bytes_bin"] = ((raw["sbytes"] + raw["dbytes"]) / 100).round()
    if group.isna().any().any():
        raise ValueError("Missing UNSW grouping values require an explicit policy")
    return list(group.itertuples(index=False, name=None))


def connected_groups(exact, near):
    """Union both relations, including bridges between their groups."""
    parent = np.arange(len(exact))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for values in (exact, near):
        seen = {}
        for i, value in enumerate(values):
            if value in seen:
                a, b = find(i), find(seen[value])
                parent[max(a, b)] = min(a, b)
            else:
                seen[value] = i
    return np.array([find(i) for i in range(len(exact))])


def choose_validation(labels, groups):
    """Pick the closest row/class 80/20 balance among 256 fixed-seed group splits."""
    best = None
    splitter = GroupShuffleSplit(n_splits=256, test_size=0.2, random_state=SEED)
    for train, validation in splitter.split(labels, labels, groups):
        if len(np.unique(labels[train])) != 2 or len(np.unique(labels[validation])) != 2:
            continue
        # Equal weight to each class's held-out fraction; no model/test scores used.
        loss = sum(abs(np.sum(labels[validation] == c) / np.sum(labels == c) - 0.2) for c in (0, 1))
        if best is None or loss < best[0]:
            best = (loss, train, validation)
    if best is None:
        raise ValueError("Cannot form two grouped splits containing both classes")
    return best[1:]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw_dir = ROOT / "data/raw/unsw_nb15"
    raw = pd.read_csv(raw_dir / "UNSW_NB15_training-set.csv", low_memory=False)
    raw_test = pd.read_csv(raw_dir / "UNSW_NB15_testing-set.csv", low_memory=False)
    source = pd.read_csv(ALIGNED / "unsw_train.csv.gz")
    official = pd.read_csv(ALIGNED / "unsw_test.csv.gz")
    for original, aligned, prefix in ((raw, source, "unsw_train"), (raw_test, official, "unsw_test")):
        expected = [f"{prefix}:UNSW_NB15_{'training' if prefix == 'unsw_train' else 'testing'}-set:{i}" for i in range(len(original))]
        if aligned.row_id.tolist() != expected:
            raise ValueError("Aligned UNSW source order/IDs changed")
        if not np.array_equal(aligned.label, original.label):
            raise ValueError("Aligned UNSW labels changed")
        expected_features = original[["dur", "spkts", "dpkts"]].to_numpy(dtype=float)
        if not np.array_equal(aligned[FEATURES].to_numpy(dtype=float), expected_features):
            raise ValueError("Aligned UNSW values differ from source")
    arr, valid = valid_feature_matrix(source)
    if not valid.all():
        raise ValueError("UNSW training now contains invalid features; review grouping policy")
    exact = keys_for(arr)
    train_patterns = set(exact)
    near = near_keys(raw)
    test_near = set(near_keys(raw_test))
    groups = connected_groups(exact, near)
    matches = set()
    # Stream external data; only retain matching training patterns in memory.
    invalid_parts = []
    for name, filename in TESTS:
        for chunk in pd.read_csv(ALIGNED / filename, chunksize=CHUNK):
            values, ok = valid_feature_matrix(chunk)
            matches.update(k for k in keys_for(values[ok]) if k in train_patterns)
            rejected = chunk.loc[~ok, META].copy()
            rejected["reason"] = "invalid_aligned_input"
            rejected["boundary"] = "input_quality"
            invalid_parts.append(rejected)
        print("Scanned", name, flush=True)
    direct_exact = np.array([k in matches for k in exact])
    direct_near = np.array([k in test_near for k in near])
    blocked = set(groups[direct_exact | direct_near])
    excluded = np.array([g in blocked for g in groups])
    eligible = np.flatnonzero(~excluded)
    tr, va = choose_validation(source.label.to_numpy()[eligible], groups[eligible])
    assignments = np.full(len(source), "excluded", dtype=object)
    assignments[eligible[tr]] = "train"
    assignments[eligible[va]] = "validation"
    rejected = source.loc[excluded, META].copy()
    reasons = np.where(direct_exact, "exact_aligned_test_match", np.where(direct_near, "near_official_test_match", "connected_to_test_matching_group"))
    rejected["reason"] = reasons[excluded]
    rejected["boundary"] = "UNSW_train_validation_vs_tests"
    exclusions = pd.concat([rejected, *invalid_parts], ignore_index=True)
    exclusions.to_csv(OUT / "exclusions.csv.gz", index=False, compression="gzip")
    source["group_id"] = [f"unsw_group:{g}" for g in groups]
    source["split"] = assignments
    source[["row_id", "group_id", "split"]].to_csv(OUT / "unsw_groups.csv.gz", index=False, compression="gzip")
    kept = source.loc[~excluded, META + ["split"]].copy()
    split_path = ROOT / "split.csv.gz"
    kept.to_csv(split_path, index=False, compression="gzip")
    # Append through one gzip text stream; all kept external rows are tests.
    import gzip
    with gzip.open(split_path, "at", encoding="utf-8", newline="") as stream:
        for _, filename in TESTS:
            for chunk in pd.read_csv(ALIGNED / filename, chunksize=CHUNK):
                _, ok = valid_feature_matrix(chunk)
                rows = chunk.loc[ok, META].copy()
                rows["split"] = "test"
                rows.to_csv(stream, header=False, index=False)
    # Verify serialized assignments against fresh reads of aligned inputs.
    saved = pd.read_csv(split_path)
    if saved.row_id.duplicated().any() or exclusions.row_id.duplicated().any():
        raise AssertionError("Row IDs must be globally unique")
    if set(saved.row_id) & set(exclusions.row_id):
        raise AssertionError("Excluded IDs occur in assignments")
    train_keys = set(np.array(exact, dtype=object)[assignments == "train"])
    val_keys = set(np.array(exact, dtype=object)[assignments == "validation"])
    checks = {"train_rows_matching_validation": sum(k in val_keys for k, s in zip(exact, assignments) if s == "train")}
    test_matches = set()
    original_rows = len(source)
    for name, filename in TESTS:
        count = 0
        for chunk in pd.read_csv(ALIGNED / filename, chunksize=CHUNK):
            original_rows += len(chunk)
            values, ok = valid_feature_matrix(chunk)
            selected = saved[saved.row_id.isin(chunk.row_id)]
            if set(selected.row_id) != set(chunk.loc[ok, "row_id"]):
                raise AssertionError("Serialized test retention differs from quality policy")
            test_matches.update(k for k in keys_for(values[ok]) if k in train_keys)
            count += len(selected)
        print("Verified", name, count, "test rows", flush=True)
    checks["train_rows_matching_any_test"] = sum(k in test_matches for k, s in zip(exact, assignments) if s == "train")
    train_near = {k for k, s in zip(near, assignments) if s == "train"}
    validation_near = {k for k, s in zip(near, assignments) if s == "validation"}
    checks["near_groups_crossing_train_validation"] = len(train_near & validation_near)
    checks["near_groups_crossing_development_official_test"] = len({k for k, s in zip(near, assignments) if s != "excluded"} & test_near)
    checks["connected_groups_crossing_train_validation"] = len(set(groups[assignments == "train"]) & set(groups[assignments == "validation"]))
    if any(checks.values()):
        raise AssertionError(checks)
    if original_rows != len(saved) + len(exclusions):
        raise AssertionError("Source rows do not reconcile")
    # Ensure saved development assignments and labels equal the chosen groups.
    pd.testing.assert_frame_equal(saved[saved.split != "test"].reset_index(drop=True), kept.reset_index(drop=True))
    counts = saved.groupby(["dataset", "split"]).label.agg(rows="size", attack="sum").reset_index()
    counts["benign"] = counts.rows - counts.attack
    # Explicit zeros for external datasets' unused development splits.
    index = pd.MultiIndex.from_product([sorted(saved.dataset.unique()), ["train", "validation", "test"]], names=["dataset", "split"])
    counts = counts.set_index(["dataset", "split"]).reindex(index, fill_value=0).reset_index()
    counts.to_csv(OUT / "split_counts.csv", index=False)
    removed = exclusions.groupby(["dataset", "reason", "boundary"]).label.agg(rows="size", attack="sum").reset_index()
    removed["benign"] = removed.rows - removed.attack
    removed.to_csv(OUT / "exclusion_counts.csv", index=False)
    summary = {"seed": SEED, "validation_target": 0.2, "candidate_group_splits": 256,
               "original_rows": original_rows, "kept_rows": len(saved), "excluded_rows": len(exclusions),
               "direct_exact_training_exclusions": int(direct_exact.sum()),
               "additional_group_training_exclusions": int((excluded & ~direct_exact).sum()),
               "eligible_development_rows": len(eligible), "eligible_connected_groups": int(len(set(groups[eligible]))),
               "largest_eligible_group": int(pd.Series(groups[eligible]).value_counts().max()),
               "actual_validation_fraction": len(va) / len(eligible), "checks": checks}
    (OUT / "leakage_checks.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = ["# Section 2(c): Final split counts", "", "Generated by `src/create_checkpoint_splits.py`; raw and aligned inputs are unchanged.", "",
        "Dataset-boundary design: only the official UNSW training source supplies training/validation; the official UNSW test and both external datasets supply test rows.", "",
        "Shared feature mapping (labels: benign=0, attack=1; collector semantics remain a limitation):\n\n| Input | UNSW-NB15 | CICIDS2017 | TON_IoT |\n|---|---|---|---|\n| duration_seconds | dur | Flow Duration / 1,000,000 | duration |\n| forward_packets | spkts | Total Fwd Packets | src_pkts |\n| backward_packets | dpkts | Total Backward Packets | dst_pkts |", "",
        "Invalid-input policy: exclude any row with a missing, nonnumeric, infinite, or negative aligned input. Zero is valid. Equality uses float64 with negative zero normalized, without rounding or labels.", "",
        "UNSW operational near-group rule: exact equality of the 39 network features other than dur, sbytes and dbytes; duration rounded to 0.01 seconds and total sbytes+dbytes rounded to nearest 100 (pandas half-to-even). IDs and target fields are excluded. This refines the broader Section 2(b) screening heuristic; it does not establish event identity or exhaust all possible near-duplicates.", "",
        "Link training-source rows whenever either their three aligned inputs or their operational near-group keys match. Exclude whole connected components touching any valid test input pattern or an official UNSW test near-group. This protects both development partitions from tests, including indirect bridges. Exclusion reasons below are mutually exclusive, in exact-match, near-match, then connected-component order.", "",
        "Split remaining components approximately 80/20 using GroupShuffleSplit seed 42. Choose the first minimum of 256 candidates by the sum, over both classes, of absolute deviation from a 20% validation fraction. No test labels or model scores select assignments.", "",
        "| Dataset | Split | Rows | Benign | Attack |", "|---|---|---:|---:|---:|"]
    for r in counts.itertuples():
        report.append(f"| {r.dataset} | {r.split} | {r.rows:,} | {r.benign:,} | {r.attack:,} |")
    report += ["", "## Exclusions", "", "| Dataset | Reason | Boundary/purpose | Rows | Benign | Attack |", "|---|---|---|---:|---:|---:|"]
    for r in removed.itertuples():
        report.append(f"| {r.dataset} | {r.reason} | {r.boundary} | {r.rows:,} | {r.benign:,} | {r.attack:,} |")
    report += ["", f"Reconciliation: {original_rows:,} original rows = {len(saved):,} kept + {len(exclusions):,} excluded. No rows moved between original datasets/official partitions.", "",
        f"Exact aligned-pattern exclusions remove {direct_exact.sum():,} training-source rows ({direct_exact.mean():.2%}); grouping excludes another {(excluded & ~direct_exact).sum():,}. This is a strict input-pattern-disjoint experiment: common behavior can match across unrelated events, so these exclusions change the training distribution and limit generalization claims.", "",
        "Within-partition duplicates are retained. CICIDS2017 and TON_IoT each occupy only test; no internal training/validation boundary exists there. Test-to-test overlap is allowed and the test sets must be reported separately.", "",
        "## Section 2(d): Post-split checks", ""]
    report.extend(f"- {name}: **{value}**." for name, value in checks.items())
    report += ["", "Related hosts/sessions and time-based independence are unavailable for prepared UNSW files: row IDs are not entity IDs. The operational near-group check cannot establish host/session independence.", "",
        "Artifacts: repository-root split.csv.gz (checker format), results/splits/unsw_groups.csv.gz, exclusions.csv.gz, split_counts.csv, exclusion_counts.csv and leakage_checks.json. Run the course checker after baseline scores.csv exists; this script does not train a model or produce scores.", ""]
    (OUT / "section_2c_final.md").write_text("\n".join(report), encoding="utf-8")
    print(counts.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
