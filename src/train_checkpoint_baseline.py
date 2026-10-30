"""Train once, choose a validation-only threshold, score tests; optionally run checker.

Run: .venv/Scripts/python.exe src/train_checkpoint_baseline.py
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, average_precision_score,
                             confusion_matrix, roc_auc_score, roc_curve)

from preview_checkpoint_splits import FEATURES, TESTS, valid_feature_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/baseline"
ALIGNED = ROOT / "data/processed/aligned"
SEED = 42


def select_threshold(y, scores, target_fpr=0.05):
    """Lowest observed score threshold satisfying empirical validation FPR cap.

    Ties are indivisible under score >= threshold. A finite value immediately
    above the maximum score represents predicting no positives when necessary.
    """
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Validation requires both classes and finite scores")
    fpr, _, thresholds = roc_curve(y, scores, drop_intermediate=False)
    thresholds[0] = np.nextafter(scores.max(), np.inf)
    return float(thresholds[fpr <= target_fpr].min())


def metrics(y, score, threshold):
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"rows": len(y), "positive_rows": int(np.sum(y)),
            "positive_fraction": float(np.mean(y)), "random_AP": float(np.mean(y)),
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "recall": float(tp / (tp + fn)), "false_positive_rate": float(fp / (fp + tn)),
            "AP": float(average_precision_score(y, score)),
            "AUROC": float(roc_auc_score(y, score)),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-checker", action="store_true", help="Also run the course checker after scoring")
    args = parser.parse_args()
    started = perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    split_path = ROOT / "split.csv.gz"
    assignments = pd.read_csv(split_path).set_index("row_id", verify_integrity=True)
    if not assignments.split.isin(["train", "validation", "test"]).all():
        raise ValueError("Unexpected split assignment")
    checks = json.loads((ROOT / "results/splits/leakage_checks.json").read_text())
    if any(checks["checks"].values()):
        raise ValueError("Saved leakage checks failed")
    source = pd.read_csv(ALIGNED / "unsw_train.csv.gz").set_index("row_id", verify_integrity=True)
    development = assignments[assignments.split != "test"]
    if set(development.dataset) != {"UNSW-NB15"}:
        raise ValueError("Only UNSW may supply development data")
    selected = source.loc[development.index]
    if not np.array_equal(selected.label, development.label):
        raise ValueError("Development labels disagree with assignments")
    _, valid = valid_feature_matrix(selected)
    if not valid.all():
        raise ValueError("Invalid development inputs")
    train = selected[development.split == "train"]
    validation = selected[development.split == "validation"]
    model = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                   random_state=SEED, n_jobs=-1)
    print(f"Training Random Forest: {len(train):,} rows, {FEATURES}", flush=True)
    timer = perf_counter()
    model.fit(train[FEATURES], train.label)
    fit_seconds = perf_counter() - timer
    print(f"Training completed in {fit_seconds:.2f} seconds", flush=True)
    attack_index = list(model.classes_).index(1)
    validation_scores = model.predict_proba(validation[FEATURES])[:, attack_index]
    threshold = select_threshold(validation.label.to_numpy(), validation_scores)
    validation_metrics = metrics(validation.label.to_numpy(), validation_scores, threshold)
    print(f"Validation threshold: {threshold!r}; FPR={validation_metrics['false_positive_rate']:.4%}", flush=True)
    if validation_metrics["false_positive_rate"] > 0.05:
        raise AssertionError("Validation threshold violates FPR cap")
    pd.DataFrame({"row_id": validation.index, "label": validation.label.to_numpy(),
                  "score": validation_scores}).to_csv(OUT / "validation_scores.csv.gz", index=False)
    joblib.dump({"model": model, "features": FEATURES, "threshold": threshold,
                 "positive_label": 1}, model_dir / "random_forest.joblib", compress=3)
    scores_path = ROOT / "scores.csv.gz"
    test_index = assignments[assignments.split == "test"]
    seen = set()
    test_metrics = {}
    timer = perf_counter()
    with gzip.open(scores_path, "wt", encoding="utf-8", newline="") as stream:
        header = True
        for name, filename in TESTS:
            labels, probabilities = [], []
            for chunk in pd.read_csv(ALIGNED / filename, chunksize=100_000):
                keep = chunk.row_id.isin(test_index.index)
                rows = chunk.loc[keep].copy()
                if rows.empty:
                    continue
                expected = test_index.loc[rows.row_id]
                if not np.array_equal(rows.label, expected.label) or not np.array_equal(rows.dataset, expected.dataset):
                    raise ValueError("Test metadata differs from assignments")
                ids = set(rows.row_id)
                if len(ids) != len(rows) or seen & ids:
                    raise ValueError("Test row scored more than once")
                seen.update(ids)
                _, valid = valid_feature_matrix(rows)
                if not valid.all():
                    raise ValueError("Invalid retained test features")
                score = model.predict_proba(rows[FEATURES])[:, attack_index]
                pd.DataFrame({"row_id": rows.row_id.to_numpy(), "test_set": name,
                              "label": rows.label.to_numpy(), "score": score}).to_csv(stream, index=False, header=header)
                header = False
                labels.append(rows.label.to_numpy())
                probabilities.append(score)
            y, score = np.concatenate(labels), np.concatenate(probabilities)
            test_metrics[name] = metrics(y, score, threshold)
            print(f"Scored {name}: {len(y):,} rows; AUROC={test_metrics[name]['AUROC']:.4f}", flush=True)
            if name == "UNSW-NB15 official test":
                fig, ax = plt.subplots(figsize=(6, 5))
                ConfusionMatrixDisplay(confusion_matrix(y, score >= threshold, labels=[0, 1]),
                                       display_labels=["Benign", "Attack"]).plot(ax=ax, colorbar=False, values_format=",")
                ax.set_title("Random Forest: UNSW-NB15 official test")
                fig.tight_layout()
                fig.savefig(OUT / "unsw_confusion_matrix.png", dpi=180)
                plt.close(fig)
    if seen != set(test_index.index):
        raise ValueError("Not every assigned test row received exactly one score")
    scoring_seconds = perf_counter() - timer
    metadata = {"classifier": "RandomForestClassifier", "parameters": model.get_params(),
                "features": FEATURES, "preprocessing": "Aligned duration/packet counts; invalid rows excluded by finalized assignments. No scaling or imputation.",
                "positive_label": 1, "seed": SEED, "threshold": threshold,
                "threshold_rule": "Lowest observed validation score threshold with empirical FPR <= 5%; ties kept together; finite above-max fallback.",
                "fit_seconds": fit_seconds, "test_scoring_and_metrics_seconds": scoring_seconds,
                "hardware": {"platform": platform.platform(), "processor": platform.processor(), "logical_cpus": os.cpu_count()},
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
                "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
                "validation": validation_metrics, "tests": test_metrics}
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (OUT / "environment.txt").write_text(freeze.stdout, encoding="utf-8")
    command = [sys.executable, str(ROOT / "check_checkpoint.py"), "--split", str(split_path),
               "--scores", str(scores_path), "--threshold", repr(threshold)]
    (OUT / "checker_command.txt").write_text(subprocess.list2cmdline(command) + "\n", encoding="utf-8")
    metadata["baseline_workflow_seconds"] = perf_counter() - started
    metadata["checker_status"] = "pending"
    if args.run_checker:
        print("Running course checker with its default bootstrap counts; this can take several minutes.", flush=True)
        timer = perf_counter()
        with (OUT / "checker_output.txt").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        metadata["checker_seconds"] = perf_counter() - timer
        metadata["checker_status"] = "completed"
    metadata["total_seconds"] = perf_counter() - started
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    report = ["# Random Forest checkpoint baseline", "", f"Training: {len(train):,} rows; validation: {len(validation):,} rows. Seed: 42.",
              f"Fit time: {fit_seconds:.2f} seconds on {metadata['hardware']}. No refit after validation.", "",
              f"Features: {', '.join(FEATURES)}. No scaling or imputation. 200 trees, min_samples_leaf=2; other settings are recorded in run_metadata.json.", "",
              f"Positive class: attack (1). Threshold: **{threshold!r}**, selected only on validation using a 5% empirical false-positive cap. Validation FPR: {validation_metrics['false_positive_rate']:.4%}; recall: {validation_metrics['recall']:.4%}.", "",
              "Point estimates below are preliminary. Run the course checker using checker_command.txt for its official metrics, 95% bootstrap intervals and warnings; use those values in the checkpoint. Checker status: " + metadata["checker_status"] + ".", "",
              "| Classifier | UNSW-NB15 official test AUROC | CICIDS2017 AUROC | TON_IoT AUROC |", "|---|---:|---:|---:|",
              "| Random Forest | " + " | ".join(f"{test_metrics[n]['AUROC']:.6f}" for n, _ in TESTS) + " |", "| XGBoost (pending) | — | — | — |", "",
              "![UNSW confusion matrix](unsw_confusion_matrix.png)", "",
              f"Figure: UNSW-NB15 official test, {test_metrics[TESTS[0][0]]['rows']:,} rows, {test_metrics[TESTS[0][0]]['positive_fraction']:.2%} attacks. At the validation-selected threshold, test recall is {test_metrics[TESTS[0][0]]['recall']:.2%} and test FPR is {test_metrics[TESTS[0][0]]['false_positive_rate']:.2%}; the validation FPR cap is not a guarantee for other data.", "",
              "Limitations: three-feature representation, collector differences, strict training-pattern exclusions, and unavailable UNSW host/session identifiers. Row-bootstrap intervals do not establish independence among repeated or related flows. Test results were not used to tune this baseline.", ""]
    for name, values in test_metrics.items():
        if values["AP"] <= values["random_AP"] or values["AUROC"] <= 0.5:
            report.append(f"Random-reference check: {name} does not beat random on AP or AUROC. Score direction is attack probability and labels agree with split.csv.gz; distribution/feature shifts are a possible explanation, not a demonstrated cause.")
        if any(values[k] is not None and values[k] >= .99 for k in ["precision", "recall", "AP", "AUROC"]):
            report.append(f"Too-good check: {name} reaches 0.99 on at least one metric. Saved split checks report zero exact training/test and training/validation matches and zero protected near-group crossings; this does not rule out all shortcuts.")
    (OUT / "baseline_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Done. Total elapsed: {metadata['total_seconds'] / 60:.2f} minutes. See results/baseline/.", flush=True)


if __name__ == "__main__":
    main()
