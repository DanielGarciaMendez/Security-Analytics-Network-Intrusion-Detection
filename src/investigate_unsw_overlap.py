"""Investigate exact UNSW-NB15 train/test feature overlap (read-only).

From the repository root on Windows:
  python src/investigate_unsw_overlap.py

Inputs: data/raw/unsw_nb15/UNSW_NB15_training-set.csv
        data/raw/unsw_nb15/UNSW_NB15_testing-set.csv
Outputs: results/audits/unsw_nb15/overlap_investigation.txt
         results/audits/unsw_nb15/overlap_examples.csv
         results/audits/unsw_nb15/conflicting_label_examples.csv
         results/audits/unsw_nb15/overlapping_row_ids.csv

This investigation never changes the original dataset files or drops any rows.
"""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "raw" / "unsw_nb15"
OUT_DIR = ROOT / "results" / "audits" / "unsw_nb15"


def pretty(value):
    return f"{int(value):,}"


def summarize_patterns(dataframe, feature_columns, prefix):
    """Exact equality on all chosen columns; count benign/attack per pattern."""
    counts = (
        dataframe.groupby(feature_columns + ["label"], dropna=False, sort=False)
        .size()
        .unstack("label", fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
        .rename(columns={0: f"{prefix}_benign", 1: f"{prefix}_attack"})
        .reset_index()
    )
    return counts


def write_report(lines):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    path = OUT_DIR / "overlap_investigation.txt"
    path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Saved report: {path.relative_to(ROOT)}")


def main():
    train_file = IN_DIR / "UNSW_NB15_training-set.csv"
    test_file = IN_DIR / "UNSW_NB15_testing-set.csv"
    for file in (train_file, test_file):
        if not file.is_file():
            sys.exit(f"ERROR: Missing CSV: {file}\nCheck that you are using the project folder structure from README.md")

    print("Reading the two original UNSW CSVs...")
    train = pd.read_csv(train_file, low_memory=False)
    test = pd.read_csv(test_file, low_memory=False)

    required = {"id", "label", "attack_cat"}
    for name, df in (("training", train), ("testing", test)):
        absent = required - set(df.columns)
        if absent:
            sys.exit(f"ERROR: {name} dataset missing columns {sorted(absent)}")
        if df.columns.duplicated().any():
            sys.exit(f"ERROR: {name} dataset has repeated column names")
        if df["label"].isna().any() or not df["label"].isin([0, 1]).all():
            sys.exit(f"ERROR: {name} dataset contains unexpected or missing binary labels")

    features = [col for col in train.columns if col not in required]
    if set(features) != set(test.columns) - required:
        sys.exit("ERROR: training and testing files have different feature columns")
    test = test[["id", *features, "attack_cat", "label"]]

    print("Grouping records by identical network-feature values (excluding IDs and answers)...")
    train_patterns = summarize_patterns(train, features, "train")
    test_patterns = summarize_patterns(test, features, "test")
    overlap = train_patterns.merge(test_patterns, on=features, how="inner", validate="one_to_one")

    overlap["train_total"] = overlap["train_benign"] + overlap["train_attack"]
    overlap["test_total"] = overlap["test_benign"] + overlap["test_attack"]
    overlap["combined_total"] = overlap["train_total"] + overlap["test_total"]
    overlap["cross_file_label_disagreement"] = (
        ((overlap["train_benign"] > 0) & (overlap["test_attack"] > 0))
        | ((overlap["train_attack"] > 0) & (overlap["test_benign"] > 0))
    )
    overlap["mixed_labels_anywhere"] = (
        ((overlap["train_benign"] + overlap["test_benign"]) > 0)
        & ((overlap["train_attack"] + overlap["test_attack"]) > 0)
    )

    print("Locating original row IDs for each overlapping feature pattern...")
    # Exact key equality using pandas' tuple MultiIndex, not approximate similarity.
    overlap_index = pd.MultiIndex.from_frame(overlap[features])
    in_train = pd.MultiIndex.from_frame(train[features]).isin(overlap_index)
    in_test = pd.MultiIndex.from_frame(test[features]).isin(overlap_index)

    if int(in_train.sum()) != int(overlap["train_total"].sum()) or int(in_test.sum()) != int(overlap["test_total"].sum()):
        raise RuntimeError("Overlap counts did not reconcile; investigate keys and missing values")

    train_overlap = train.loc[in_train]
    test_overlap = test.loc[in_test]

    # Keep the pattern examples short enough to inspect in VS Code/Excel.
    display_features = [
        x for x in ["dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl"]
        if x in features
    ]
    count_cols = [
        "train_benign", "train_attack", "test_benign", "test_attack",
        "train_total", "test_total", "combined_total", "cross_file_label_disagreement",
    ]
    largest = overlap.sort_values("combined_total", ascending=False)
    example_columns = display_features + count_cols
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    largest.head(25)[example_columns].to_csv(OUT_DIR / "overlap_examples.csv", index=False)
    overlap.loc[overlap["cross_file_label_disagreement"]].sort_values(
        "combined_total", ascending=False
    ).head(25)[example_columns].to_csv(OUT_DIR / "conflicting_label_examples.csv", index=False)

    def row_ids(df, selection, partition):
        out = df.loc[selection, ["id", "label", "attack_cat"]].copy()
        out.insert(0, "row_id", partition + ":" + out["id"].astype(str))
        out.insert(1, "partition", partition)
        return out.drop(columns=["id"])

    pd.concat([
        row_ids(train, in_train, "unsw_train"),
        row_ids(test, in_test, "unsw_test"),
    ], ignore_index=True).to_csv(OUT_DIR / "overlapping_row_ids.csv", index=False)

    cross_disagree = overlap["cross_file_label_disagreement"]
    inconsistent_train = (train_patterns["train_benign"] > 0) & (train_patterns["train_attack"] > 0)
    inconsistent_test = (test_patterns["test_benign"] > 0) & (test_patterns["test_attack"] > 0)

    lines = [
        "=" * 75,
        "UNSW-NB15 | TRAIN / OFFICIAL-TEST OVERLAP INVESTIGATION",
        "=" * 75,
        "",
        "PURPOSE: Find exact feature-pattern overlap and label disagreements.",
        "ORIGINAL DATA: No records were deleted, modified, or split by this script.",
        "",
        "1. INPUT DATA",
        f"   Training records:                  {pretty(len(train))}",
        f"   Official testing records:          {pretty(len(test))}",
        f"   Network features compared:         {pretty(len(features))}",
        "   Excluded from equality:           id, label, attack_cat",
        "",
        "2. WITHIN-FILE DUPLICATES (NOT THE SAME AS CROSS-FILE OVERLAP)",
        f"   Train duplicates excluding id:     {pretty(train.drop(columns=['id']).duplicated().sum())}",
        f"   Test duplicates excluding id:      {pretty(test.drop(columns=['id']).duplicated().sum())}",
        f"   Train duplicate network features:  {pretty(train[features].duplicated().sum())}",
        f"   Test duplicate network features:   {pretty(test[features].duplicated().sum())}",
        "",
        "3. TRAIN / OFFICIAL-TEST OVERLAP",
        f"   Distinct feature patterns shared:  {pretty(len(overlap))}",
        f"   Training rows matching test:       {pretty(in_train.sum())} ({in_train.mean():.2%})",
        f"   Testing rows matching training:    {pretty(in_test.sum())} ({in_test.mean():.2%})",
        "   Each count is a count of ROWS, not unique patterns.",
        "   Do NOT add train and test counts together as unique duplicates.",
        "",
        "4. LABEL CONSISTENCY",
        f"   Feature patterns with both labels anywhere:  {pretty(overlap['mixed_labels_anywhere'].sum())}",
        f"   Patterns with opposite labels across files: {pretty(cross_disagree.sum())}",
        f"   Train feature patterns with both labels:    {pretty(inconsistent_train.sum())}",
        f"   Test feature patterns with both labels:     {pretty(inconsistent_test.sum())}",
        "   Opposite labels do not automatically mean either dataset is wrong.",
        "",
        "5. LABEL COUNTS AMONG OVERLAPPING ROWS",
        f"   Training / benign: {pretty((train_overlap['label'] == 0).sum())}",
        f"   Training / attack: {pretty((train_overlap['label'] == 1).sum())}",
        f"   Testing  / benign: {pretty((test_overlap['label'] == 0).sum())}",
        f"   Testing  / attack: {pretty((test_overlap['label'] == 1).sum())}",
        "",
        "6. ATTACK CATEGORIES IN OVERLAPPING ROWS (TRAIN)",
    ]
    for category, count in train_overlap["attack_cat"].astype(str).value_counts().items():
        lines.append(f"   {category:<24} {pretty(count):>10}")
    lines += ["", "7. ATTACK CATEGORIES IN OVERLAPPING ROWS (OFFICIAL TEST)"]
    for category, count in test_overlap["attack_cat"].astype(str).value_counts().items():
        lines.append(f"   {category:<24} {pretty(count):>10}")
    lines += [
        "", "8. INSPECTABLE EVIDENCE",
        "   overlap_examples.csv:              25 most frequent shared feature patterns",
        "   conflicting_label_examples.csv:    up to 25 patterns with opposite labels across files",
        "   overlapping_row_ids.csv:           original row IDs affected (both partitions)",
        "", "9. INTERPRETATION AND LIMITATIONS",
        "   These are exact matches of all available NETWORK FEATURE values.",
        "   Matching features do NOT prove the records are the same real event.",
        "   A matching pattern in train and test can inflate held-out evaluation.",
        "   No cleaning decision has yet been made or implemented.",
        "   The check MUST be repeated after choosing final aligned model features.",
        "   Near-duplicate detection and group-aware train/validation splitting are separate tasks.",
    ]
    write_report(lines)
    for filename in ("overlap_examples.csv", "conflicting_label_examples.csv", "overlapping_row_ids.csv"):
        print(f"Saved: {(OUT_DIR / filename).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
