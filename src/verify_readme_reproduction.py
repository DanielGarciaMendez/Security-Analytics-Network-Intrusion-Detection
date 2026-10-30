"""Test the README rebuild in an isolated source/output workspace.

Reuses the installed environment and read-only raw inputs (hard links); does not
install packages, download data, or overwrite existing research outputs.
The pipeline scripts only read raw inputs. Workspace is retained for inspection.
"""
from importlib.metadata import version
import argparse
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/reproduction"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-only", action="store_true", help="Compare a retained, completed workspace without rebuilding")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    workspace = OUT / "workspace"
    if workspace.exists() and not args.compare_only:
        raise FileExistsError("Workspace already exists; retain it or move it before a new verification run")
    for line in (ROOT / "requirements-lock.txt").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            package, pinned = line.split("==")
            if version(package) != pinned:
                raise ValueError(f"Installed {package} does not match lock file")
    if not args.compare_only:
        workspace.mkdir()
        shutil.copytree(ROOT / "src", workspace / "src", ignore=shutil.ignore_patterns("__pycache__"))
        for name in ["README.md", "check_checkpoint.py", "requirements.txt", "requirements-lock.txt"]:
            shutil.copy2(ROOT / name, workspace / name)
    source_hash = hashlib.sha256()
    for path in sorted((workspace / "src").glob("*.py")):
        source_hash.update(path.name.encode()); source_hash.update(path.read_bytes())
    raw_files = list((ROOT / "data/raw").rglob("*.csv"))
    if len(raw_files) != 11:
        raise ValueError("Expected 11 raw input CSV files")
    if not args.compare_only:
        for path in raw_files:
            destination = workspace / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, destination)
    command = [sys.executable, "-u", "src/run_checkpoint.py"]
    start = perf_counter()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if not args.compare_only:
        print("Rebuilding from raw inputs in", workspace, flush=True)
        with (OUT / "run.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=workspace, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError("README rebuild failed; see results/reproduction/run.log")
    # CSV comparisons ignore gzip timestamps, which differ between reruns.
    old = pd.read_csv(ROOT / "split.csv.gz")
    new = pd.read_csv(workspace / "split.csv.gz")
    pd.testing.assert_frame_equal(old, new)
    del old, new
    score_rows = 0
    largest_difference = 0.0
    changed_predictions = 0
    original_metadata = json.loads((ROOT / "results/baseline/run_metadata.json").read_text())
    rebuilt_metadata = json.loads((workspace / "results/baseline/run_metadata.json").read_text())
    old_threshold = original_metadata["threshold"]
    new_threshold = rebuilt_metadata["threshold"]
    if abs(old_threshold - new_threshold) > 1e-12:
        raise AssertionError("Validation threshold changed materially")
    first = pd.read_csv(ROOT / "scores.csv.gz", chunksize=100_000)
    second = pd.read_csv(workspace / "scores.csv.gz", chunksize=100_000)
    for old, new in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(old.drop(columns="score"), new.drop(columns="score"))
        np.testing.assert_allclose(old.score, new.score, rtol=0, atol=1e-12)
        largest_difference = max(largest_difference, float(np.max(np.abs(old.score - new.score))))
        changed_predictions += int(np.sum((old.score >= old_threshold) != (new.score >= new_threshold)))
        score_rows += len(new)
    if changed_predictions:
        raise AssertionError("Thresholded test predictions changed")
    tree_hashes = []
    for model_path in (ROOT / "models/random_forest.joblib", workspace / "models/random_forest.joblib"):
        model = joblib.load(model_path)["model"]
        digest = hashlib.sha256()
        for estimator in model.estimators_:
            for field in ("children_left", "children_right", "feature", "threshold", "value", "n_node_samples", "weighted_n_node_samples"):
                digest.update(getattr(estimator.tree_, field).tobytes())
        tree_hashes.append(digest.hexdigest())
        del model
    if tree_hashes[0] != tree_hashes[1]:
        raise AssertionError("Rebuilt fitted trees differ from the saved model")
    old_checker = json.loads((ROOT / "results/checkpoint/checker_metrics.json").read_text())
    new_checker = json.loads((workspace / "results/checkpoint/checker_metrics.json").read_text())
    printed_identical = old_checker == new_checker
    summary = {"status": "passed" if printed_identical else "completed_with_floating_point_differences",
               "verification_elapsed_seconds": perf_counter() - start,
               "comparison_only": args.compare_only,
               "command": "python src/run_checkpoint.py", "python": sys.version,
               "environment": "Existing environment; all installed versions match requirements-lock.txt. No fresh installation tested.",
               "inputs": "11 existing raw CSVs linked read-only by pipeline convention; no processed data or prior outputs copied",
               "split_assignments_identical": True, "test_score_rows": score_rows,
               "maximum_absolute_score_difference": largest_difference,
               "test_predictions_changed": changed_predictions,
               "fitted_tree_arrays_identical": True, "fitted_tree_sha256": tree_hashes[0],
               "checker_metrics_and_intervals_identical": printed_identical,
               "original_threshold": old_threshold, "rebuilt_threshold": new_threshold,
               "original_AUROC": {k: v["AUROC"] for k, v in original_metadata["tests"].items()},
               "rebuilt_AUROC": {k: v["AUROC"] for k, v in rebuilt_metadata["tests"].items()},
               "numerical_note": "Threaded tree-probability sums can differ at floating-point precision. Tiny differences break score ties and can change rank metrics. Original and rebuilt checker outputs are retained; no scores are rounded to hide differences.",
               "source_snapshot_sha256": source_hash.hexdigest(),
               "limitations": "Not a fresh Git clone, not an independent download, and not a new Python environment."}
    (OUT / "verification.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
