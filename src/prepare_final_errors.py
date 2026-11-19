"""Sample and analyze errors from the saved UNSW-NB15 Random Forest scores."""

import csv
import gzip
import json
import math
import random
from collections import Counter
from prepare_final_scores import FINAL_COLUMNS, MODEL, ROOT, TEST_VALUES, read_rows, staged_file


DATASET = "UNSW-NB15"
VALUE = TEST_VALUES[DATASET]
THRESHOLD = 0.9426192528735632
SEED = 42
SAMPLE_SIZE = 10
RAW = ROOT / "data/raw/unsw_nb15/UNSW_NB15_testing-set.csv"
ALIGNED = ROOT / "data/processed/aligned/unsw_test.csv.gz"
PREFIX = f"unsw_test:{RAW.stem}:"
FEATURES = ["duration_seconds", "forward_packets", "backward_packets"]
ERROR_COLUMNS = ["row_id", "value", "column", "category"]
FP, FN = (0, 1), (1, 0)
CATEGORIES = {
    "Single-packet with no reply": "Exactly one forward packet and zero backward packets are recorded.",
    "Multiple packets with no reply": "At least two forward packets and zero backward packets are recorded.",
    "Reverse-heavy bidirectional": "Both directions have packets and the backward count exceeds the forward count.",
    "Forward-heavy or balanced bidirectional": "Both directions have packets and the forward count is at least the backward count.",
}


def category(features):
    _, forward, backward = features
    if forward < 1 or backward < 0:
        raise ValueError(f"Unsupported sampled packet counts: {forward}, {backward}")
    if backward == 0:
        return "Single-packet with no reply" if forward == 1 else "Multiple packets with no reply"
    if backward > forward:
        return "Reverse-heavy bidirectional"
    return "Forward-heavy or balanced bidirectional"


def load_predictions():
    metadata = json.loads((ROOT / "results/baseline/run_metadata.json").read_text(encoding="utf-8"))
    if (metadata["threshold"] != THRESHOLD or metadata["features"] != FEATURES
            or metadata["positive_label"] != 1):
        raise ValueError("Saved model metadata does not match the analysis specification")
    predictions = {}
    counts = Counter()
    with gzip.open(ROOT / "scores.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row_id, value, column, label, score, threshold in read_rows(stream, FINAL_COLUMNS):
            if (value, column) != (VALUE, MODEL):
                continue
            if (not row_id.startswith(PREFIX) or row_id in predictions or label not in {"0", "1"}
                    or float(threshold) != THRESHOLD or not math.isfinite(float(score))):
                raise ValueError(f"Invalid prediction in the selected cell: {row_id}")
            pair = int(label), int(float(score) >= THRESHOLD)
            predictions[row_id] = (pair, score)
            counts[pair] += 1
    if {label for label, _ in counts} != {0, 1} or not (counts[FP] or counts[FN]):
        raise ValueError("The selected cell must contain both classes and misclassifications")
    return predictions, counts


def verify_split(predictions):
    seen = set()
    with gzip.open(ROOT / "split.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            row_id = row["row_id"]
            if row_id not in predictions:
                continue
            if (row_id in seen or row["dataset"] != DATASET or row["split"] != "test"
                    or int(row["label"]) != predictions[row_id][0][0]):
                raise ValueError(f"Split does not match prediction for {row_id}")
            seen.add(row_id)
    if seen != predictions.keys():
        raise ValueError("Some scored rows are missing from the split")


def draw_sample(predictions):
    rng = random.Random(SEED)
    sample = []
    for pair in (FP, FN):
        pool = sorted(row_id for row_id, (labels, _) in predictions.items() if labels == pair)
        sample.extend(rng.sample(pool, min(SAMPLE_SIZE, len(pool))))
    return sample


def join_flows(sample, predictions):
    aligned = {}
    feature_labels = Counter()
    seen = set()
    sample_ids = set(sample)
    with gzip.open(ALIGNED, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            row_id = row["row_id"]
            if row_id not in predictions:
                continue
            label = int(row["label"])
            if row_id in seen or label != predictions[row_id][0][0] or row["dataset"] != DATASET:
                raise ValueError(f"Aligned data does not match prediction for {row_id}")
            seen.add(row_id)
            features = (float(row[FEATURES[0]]), int(row[FEATURES[1]]), int(row[FEATURES[2]]))
            feature_labels[features, label] += 1
            if row_id in sample_ids:
                aligned[row_id] = features
    if seen != predictions.keys():
        raise ValueError("Some scored rows are missing from the aligned flows")

    flows = {}
    with RAW.open(encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            row_id = f"{PREFIX}{index}"
            if row_id not in sample_ids:
                continue
            label = predictions[row_id][0][0]
            features = aligned[row_id]
            if (int(row["label"]) != label
                    or (float(row["dur"]), int(row["spkts"]), int(row["dpkts"])) != features):
                raise ValueError(f"Original flow does not match aligned input for {row_id}")
            flows[row_id] = (features, row, feature_labels[features, 1 - label])
    if flows.keys() != sample_ids:
        raise ValueError("Some sampled IDs are missing from the original flows")
    return flows


def validate_errors(path, sample, predictions, assignments, counts):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(read_rows(stream, ERROR_COLUMNS))
    seen = set()
    sampled_counts = Counter()
    for row_id, value, column, assigned in rows:
        if (row_id in seen or row_id not in predictions or (value, column) != (VALUE, MODEL)
                or assigned not in CATEGORIES or assigned != assignments.get(row_id)):
            raise ValueError(f"Invalid or duplicate error row: {row_id}")
        pair = predictions[row_id][0]
        if pair not in (FP, FN):
            raise ValueError(f"Sampled row is correctly classified: {row_id}")
        seen.add(row_id)
        sampled_counts[pair] += 1
    if [row[0] for row in rows] != sample or any(
        sampled_counts[pair] != min(SAMPLE_SIZE, counts[pair]) for pair in (FP, FN)
    ):
        raise ValueError("Published rows do not match the seeded draw or required sample sizes")
    if len(set(assignments.values())) < 2:
        raise ValueError("Expected at least two observed flow categories")
    return sampled_counts


def make_report(sample, predictions, counts, sampled_counts, flows, assignments):
    category_counts = Counter((assignments[row_id], predictions[row_id][0]) for row_id in sample)
    examples = {}
    for row_id in sample:
        examples.setdefault(assignments[row_id], row_id)
    total = len(sample)
    opposite_matches = sum(flows[row_id][2] > 0 for row_id in sample)
    report = [
        "# Section 4. Error analysis", "",
        "## A. Sampling methodology", "",
        f"The selected cell is **{VALUE} / {MODEL}**. It contains {counts[(0, 0)] + counts[FP]:,} "
        f"benign flows (0) and {counts[(1, 1)] + counts[FN]:,} attack flows (1). "
        f"A score at or above the saved threshold `{THRESHOLD}` predicts attack.", "",
        f"The cell has **{counts[FP]:,} false positives** and **{counts[FN]:,} false negatives**. "
        f"I sorted each error pool by row ID, then used one Python `random.Random({SEED})` instance "
        "to sample without replacement, drawing false positives first. The draw yielded "
        f"{sampled_counts[FP]} false positives and {sampled_counts[FN]} false negatives ({total} total).", "",
        "I matched each sampled ID to the split and aligned features, then used the zero-based "
        "source row index encoded in the ID (excluding the header) to locate its raw flow. "
        "Raw labels, duration, and packet counts matched the aligned records.", "",
        "## B. Error category summary", "",
        "First check for zero backward packets and separate single-packet from multi-packet flows. "
        "For bidirectional flows, compare packet counts and group ties with forward-heavy flows. "
        "These rules are mutually exclusive and use the same criteria for both error types.", "",
        "| Category | One-sentence rule | False positives | False negatives | Example row ID |",
        "|---|---|---:|---:|---|",
    ]
    for name, rule in CATEGORIES.items():
        if name in examples:
            report.append(f"| {name} | {rule} | {category_counts[name, FP]} | "
                          f"{category_counts[name, FN]} | `{examples[name]}` |")
    report.extend([
        "", "## C. Individual sampled errors", "",
        "Every row uses the threshold in Section A. Opposite-label matches count scored rows "
        "in this cell with exactly the same duration and two packet counts. Protocol, service, "
        "state, bytes, and attack family come from the original dataset. A dash in the service "
        "column is the source value. Packet and byte counts are shown as forward / backward.", "",
        "| Row ID | Actual | Predicted | Score | Duration (s) | Packets | Protocol / service / state |"
        " Bytes | Source attack family | Opposite-label matches | Category |",
        "|---|---:|---:|---:|---:|---|---|---|---|---:|---|",
    ])
    for row_id in sample:
        (actual, predicted), score = predictions[row_id]
        (duration, forward, backward), raw, opposite = flows[row_id]
        report.append(
            f"| `{row_id}` | {actual} | {predicted} | {score} | {duration} | {forward} / {backward} | "
            f"{raw['proto']} / {raw['service']} / {raw['state']} | {raw['sbytes']} / {raw['dbytes']} | "
            f"{raw['attack_cat']} | {opposite:,} | {assignments[row_id]} |"
        )
    report.extend([
        "", "## D. Interpretation", "",
        "Three sampled false positives were benign ARP flows with zero recorded duration and a 1 / 0 "
        "packet count. They scored 1.0. Seven sampled false negatives had two forward packets and "
        "zero replies, with durations from 7 to 35 microseconds. They spanned Generic and Fuzzers "
        "labels and all scored 0.7810030247522117, below the cutoff.", "",
        f"Across the sample, {opposite_matches} of {total} flows shared all three model inputs with "
        "at least one scored flow of the opposite label. The Random Forest receives identical inputs "
        "for each such pair, so its three-feature representation cannot distinguish those examples.", "",
        "The bidirectional sample also shows why packet direction alone offers an incomplete explanation. "
        "Seven benign TCP FIN flows with forward packet majorities scored above the threshold, while "
        "the bidirectional attack examples included balanced, forward-heavy, and reverse-heavy flows below "
        "it. The reverse-heavy POP3 flow had 114 forward and 656 backward packets and scored about 0.9131. "
        "Its packet imbalance and source family describe the flow, but the sample provides no evidence "
        "about which model decisions produced that particular score.", "",
        "Protocol, service, state, bytes, and attack family provide context only. The model used duration "
        "and the two packet counts. There is no packet-level evidence here to establish a root cause. "
        "The small sample deliberately balances error types within one test cell, so the category counts "
        "describe these sampled errors and do not estimate prevalence across the full test set.", "",
    ])
    return "\n".join(report)


def main():
    predictions, counts = load_predictions()
    verify_split(predictions)
    sample = draw_sample(predictions)
    flows = join_flows(sample, predictions)
    assignments = {row_id: category(flows[row_id][0]) for row_id in sample}
    output = ROOT / "results/final_checkpoint"
    output.mkdir(parents=True, exist_ok=True)
    errors_path = ROOT / "errors.csv"
    report_path = output / "error_analysis.md"
    with staged_file(errors_path) as errors, staged_file(report_path) as staged_report:
        with errors.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(ERROR_COLUMNS)
            writer.writerows((row_id, VALUE, MODEL, assignments[row_id]) for row_id in sample)
        sampled_counts = validate_errors(errors, sample, predictions, assignments, counts)
        report = make_report(sample, predictions, counts, sampled_counts, flows, assignments)
        staged_report.write_text(report, encoding="utf-8")
        staged_report.replace(report_path)
        errors.replace(errors_path)
    print(f"Available errors: {counts[FP]:,} FP, {counts[FN]:,} FN")
    print(f"Validated sample: {sampled_counts[FP]} FP, {sampled_counts[FN]} FN; "
          f"{len(set(assignments.values()))} categories")


if __name__ == "__main__":
    main()
