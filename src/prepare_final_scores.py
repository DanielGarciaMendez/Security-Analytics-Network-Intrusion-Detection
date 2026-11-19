"""Convert the saved Random Forest predictions for the final checkpoint."""

import csv
import gzip
import io
import json
import math
import os
import shutil
from collections import Counter
from contextlib import contextmanager
from itertools import zip_longest
from pathlib import Path
from tempfile import mkstemp


ROOT = Path(__file__).resolve().parents[1]
LEGACY_SCORES = "scores_legacy.csv.gz"
LEGACY_COLUMNS = ["row_id", "test_set", "label", "score"]
FINAL_COLUMNS = ["row_id", "value", "column", "label", "score", "threshold"]
MODEL = "Random Forest"
TEST_VALUES = {
    "UNSW-NB15": "UNSW-NB15 official test",
    "CICIDS2017": "CICIDS2017",
    "TON_IoT": "TON_IoT",
}


@contextmanager
def staged_file(destination):
    # A sibling file inherits the destination directory's permissions on Windows.
    descriptor, name = mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(descriptor)
    path = Path(name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def read_rows(stream, columns):
    reader = csv.reader(stream, strict=True)
    if next(reader, None) != columns:
        raise ValueError(f"Expected columns {columns} in {stream.name}")
    return reader


def test_assignments(path):
    assignments = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream, strict=True)
        if sorted(rows.fieldnames or []) != ["dataset", "label", "row_id", "split"]:
            raise ValueError(f"Unexpected split columns in {path}")
        for row in rows:
            row_id, dataset, label, split = (
                row["row_id"], row["dataset"], row["label"], row["split"]
            )
            if (None in row or not row_id or not dataset or label not in {"0", "1"}
                    or split not in {"train", "validation", "test"}):
                raise ValueError(f"Invalid split record for {row_id!r}")
            if row_id in assignments:
                raise ValueError(f"Duplicate split row ID {row_id!r}")
            assignments[row_id] = (TEST_VALUES[dataset], label) if split == "test" else None
    return {row_id: entry for row_id, entry in assignments.items() if entry is not None}


def write_candidate(source, candidate, threshold):
    with gzip.open(source, "rt", encoding="utf-8", newline="") as original:
        rows = read_rows(original, LEGACY_COLUMNS)
        with gzip.GzipFile(filename="", fileobj=candidate, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(FINAL_COLUMNS)
                for row_id, value, label, score in rows:
                    writer.writerow([row_id, value, MODEL, label, score, threshold])


def verify(source, candidate, assignments, threshold):
    counts = Counter()
    with (gzip.open(source, "rt", encoding="utf-8", newline="") as original,
          gzip.open(candidate, "rt", encoding="utf-8", newline="") as converted):
        old_rows = read_rows(original, LEGACY_COLUMNS)
        new_rows = read_rows(converted, FINAL_COLUMNS)
        for old, new in zip_longest(old_rows, new_rows):
            if old is None or new is None:
                raise ValueError("Original and converted prediction counts differ")
            row_id, value, column, label, score, saved_threshold = new
            if [row_id, value, label, score] != old:
                raise ValueError(f"Original prediction changed for {row_id!r}")
            if column != MODEL or saved_threshold != threshold:
                raise ValueError(f"Unexpected classifier or threshold for {row_id!r}")
            if not math.isfinite(float(score)):
                raise ValueError(f"Nonfinite score for {row_id!r}")
            # Each test ID is used once, so each (row_id, value, column) is unique.
            if assignments.pop(row_id, None) != (value, label):
                raise ValueError(f"Duplicate, missing, or mismatched test assignment for {row_id!r}")
            counts[value] += 1
    if assignments:
        raise ValueError(f"{len(assignments):,} test rows are missing predictions")
    if set(counts) != set(TEST_VALUES.values()):
        raise ValueError("Expected predictions for all three test datasets")
    return counts


def main():
    scores = ROOT / "scores.csv.gz"
    backup = ROOT / LEGACY_SCORES
    final_scores = ROOT / "final_scores.csv.gz"
    metadata_path = ROOT / "results/baseline/run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    threshold = metadata["threshold"]
    if type(threshold) not in (int, float) or not math.isfinite(threshold):
        raise ValueError("Random Forest metadata must provide a finite numeric threshold")
    threshold = str(threshold)
    source = backup
    if not source.exists():
        source = scores if scores.exists() else metadata_path.with_name("test_scores_legacy.csv.gz")
    assignments = test_assignments(ROOT / "split.csv.gz")

    with staged_file(final_scores) as candidate, staged_file(scores) as replacement:
        with candidate.open("wb") as stream:
            write_candidate(source, stream, threshold)
        counts = verify(source, candidate, assignments, threshold)
        # Keep the exact original gzip bytes before replacing the root path.
        if source != backup:
            os.link(source, backup)
        shutil.copyfile(candidate, replacement)
        os.replace(candidate, final_scores)
        os.replace(replacement, scores)

    print(f"Validated and published {sum(counts.values()):,} predictions at threshold {threshold}")
    for value, count in counts.items():
        print(f"  {value}: {count:,}")


if __name__ == "__main__":
    main()
