"""Build checkpoint Sections 3-5 from completed course-checker output.

No models are fitted and no scores/intervals are recomputed here.
"""
from pathlib import Path
import json
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/checkpoint"
BASE = ROOT / "results/baseline"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = (BASE / "checker_output.txt").read_text(encoding="utf-8")
    if not re.search(r"\nWarnings\n-+\n", log):
        raise ValueError("Course checker has not completed; do not report partial intervals")
    meta = json.loads((BASE / "run_metadata.json").read_text())
    sections = re.split(r"\nTest set: ", log)[1:]
    results = {}
    for block in sections:
        name = block.splitlines()[0]
        values = {}
        for metric in ["precision", "recall", "false-positive rate", "AP", "AUROC"]:
            match = re.search(r"^\s+" + re.escape(metric) + r"\s+(\d+\.\d+) \[(\d+\.\d+), (\d+\.\d+)\]", block, re.M)
            if not match:
                raise ValueError(f"Missing checker interval for {name}: {metric}")
            values[metric] = {"value": float(match[1]), "lower": float(match[2]), "upper": float(match[3]),
                              "printed": f"{match[1]} [{match[2]}, {match[3]}]"}
        for key, pattern in [("rows", r"test rows\s+(\d+)"), ("positive_rows", r"positive rows\s+(\d+)"),
                             ("positive_percent", r"positive rows\s+\d+ \(([\d.]+)%\)"),
                             ("random_AP", r"random-scorer AP\s+([\d.]+)")]:
            values[key] = re.search(pattern, block)[1]
        results[name] = values
    if len(results) != 3:
        raise ValueError("Expected all three test sets")
    warnings = re.split(r"\nWarnings\n-+\n", log)[1].strip()
    half_width = max((v["AUROC"]["upper"] - v["AUROC"]["lower"]) / 2 for v in results.values())
    margin = max(.05, (int(half_width * 1000) + 1) / 1000)
    measured_rf = meta["fit_seconds"] / 60
    scoring = meta["test_scoring_and_metrics_seconds"] / 60
    checker = meta.get("checker_seconds", 0) / 60
    compute_estimate = 11 * measured_rf + 2 * scoring + 2 * checker
    lines = ["# Checkpoint results and finish plan", "", "Daniel Garcia Mendez | COSC 4371 | Project 11 | Prepared October 29, 2026", "",
        "This document supplies Sections 3-5 for the checkpoint. Keep the completed dataset citations in your existing document; Section 2(b) is results/audits/checkpoint_2b/audit_table.md and final Sections 2(c)-2(d) evidence is results/splits/section_2c_final.md.", "",
        "## 3. Baseline and metrics", "", "### (a) Pipeline", "",
        "Map labels to benign=0 and attack=1; align duration_seconds, forward_packets and backward_packets; convert CICIDS Flow Duration from microseconds to seconds; exclude invalid input rows and apply finalized split assignments. Train only on 68,048 UNSW rows. There is no scaling or imputation. Validation has 17,018 rows; no refit includes validation.", "",
        "RandomForestClassifier: n_estimators=200, criterion=gini, max_depth=None, min_samples_split=2, min_samples_leaf=2, max_features=sqrt, bootstrap=True, class_weight=None, random_state=42, n_jobs=-1. Other defaults and full settings are recorded in results/baseline/run_metadata.json.", "",
        f"Measured fit time: {meta['fit_seconds']:.2f} seconds. Scoring/point-metrics time: {meta['test_scoring_and_metrics_seconds']:.2f} seconds. Hardware: {meta['hardware']['processor']}, {meta['hardware']['logical_cpus']} logical CPUs; {meta['hardware']['platform']}. Training/scoring workflow: {meta.get('baseline_workflow_seconds', meta['total_seconds']):.2f} seconds, excluding the subsequent course-checker run.", "",
        "### (b) Threshold", "",
        f"Positive class: attack (1). Predict attack when score >= {meta['threshold']!r}. Select the lowest observed validation score threshold whose empirical validation false-positive rate is at most 5%; score ties remain together. Threshold selection uses validation only. Measured validation FPR: {meta['validation']['false_positive_rate']:.4%}; recall: {meta['validation']['recall']:.4%}. The same model and threshold score all test datasets.", "",
        "### (c) Checker metrics", "", "Values below are copied from the unchanged course checker; brackets contain 95% row-bootstrap intervals. Default bootstrap seed=0; 1,000 UNSW resamples and 200 per external dataset. The full output is results/baseline/checker_output.txt.", "",
        "| Test set | Test rows | Positive rows (%) | Random AP |", "|---|---:|---:|---:|"]
    for name, r in results.items():
        lines.append(f"| {name} | {int(r['rows']):,} | {int(r['positive_rows']):,} ({r['positive_percent']}%) | {r['random_AP']} |")
    lines += ["", "| Test set | Precision [95% CI] | Recall [95% CI] | FPR [95% CI] |", "|---|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['precision']['printed']} | {r['recall']['printed']} | {r['false-positive rate']['printed']} |")
    lines += ["", "| Test set | AP [95% CI] | AUROC [95% CI] |", "|---|---|---|"]
    for name, r in results.items():
        lines.append(f"| {name} | {r['AP']['printed']} | {r['AUROC']['printed']} |")
    lines += ["", "### (d) Random reference and interpretation", ""]
    for name, r in results.items():
        beats = r['AP']['value'] > float(r['random_AP']) and r['AUROC']['value'] > .5
        lines.append(f"{name}: {'AP exceeds its random-scorer reference and AUROC exceeds 0.5.' if beats else 'At least one metric fails its random reference; feature/collection shifts are a possible explanation.'}")
    rf = meta['tests']['UNSW-NB15 official test']['AUROC']
    lines += ["", f"AUROC drops from UNSW by {rf-meta['tests']['CICIDS2017']['AUROC']:.4f} on CICIDS2017 and {rf-meta['tests']['TON_IoT']['AUROC']:.4f} on TON_IoT. External ranking is close to random despite above-random point estimates. This three-feature baseline does not demonstrate reliable cross-environment transfer. Collector differences and removal of 51.49% of original UNSW training-source rows limit attribution to environment alone.", "",
        "Precision and AP depend on prevalence; AUROC is the primary comparison. A validation FPR cap does not constrain test FPR. Repeated flows may violate row-bootstrap independence, so the checker's intervals do not quantify every source of uncertainty.", "",
        "### (e) Too-good check and warnings", "", "None of precision, recall, AP or AUROC reaches 0.99 on any test set. Final split checks report zero exact training/test and training/validation matches and zero protected near-group crossings. Host/session independence cannot be checked with the prepared UNSW identifiers.", "", "Checker warnings:", "", warnings, "",
        "### (f) Reproducibility", "",
        "Repository: https://github.com/DanielGarciaMendez/Security-Analytics-Network-Intrusion-Detection", "",
        f"Python {meta['versions']['python']}; exact installed packages are recorded in results/baseline/environment.txt. Seeds: split/model=42, course bootstrap=0. Data download locations and file paths are documented in README.md; raw datasets are not committed.", "",
        "One command from repository root after installing dependencies and placing raw downloads:", "", "```powershell", ".\\.venv\\Scripts\\python.exe src\\run_checkpoint.py", "```", "",
        "It rebuilds the audit, aligned inputs, assignments, baseline scores, checker output and this report. This orchestration command is provided for reproduction; a full fresh-clone execution has not been performed in this session. Existing training/scoring and checker runs were executed separately. Source/score artifacts must be committed and pushed before the repository link contains the latest work.", "",
        "## 4. Figure", "", "![Random Forest confusion matrix](../baseline/unsw_confusion_matrix.png)", "",
        "Figure: Random Forest on the UNSW-NB15 official test set; 82,332 rows and 55.06% attacks. Cells show counts; rows are true labels and columns are predicted labels.", "",
        "At the validation-selected threshold, the baseline detects 20.84% of UNSW test attacks at a 10.92% false-positive rate, showing that even source-dataset operating performance differs from validation.", "",
        "## 5. Plan to finish", "", "### (a) Required elements", "",
        "Proposed dates based on the work remaining on October 29, 2026; these are planning targets, not an assumed course deadline. Scope follows the available proposal (Random Forest and XGBoost); reconcile any additional instructor-assigned requirements separately.", "",
        "| Required element | Status | Completion/target date |", "|---|---|---|",
        "| Three datasets, binary labels and shared feature mapping | done | October 28, 2026 |",
        "| Audit, exclusions, group-aware splits and leakage checks | done | October 28, 2026 |",
        "| Recent dataset-paper citations | done (user-confirmed) | October 29, 2026 or earlier |",
        "| Random Forest training, validation threshold and three test sets | done | October 28, 2026 |",
        "| Course checker, intervals, figure and results interpretation | done | October 29, 2026 |",
        "| XGBoost baseline on the same features and assignments | not started | October 30, 2026 |",
        "| XGBoost validation-only threshold, test scores and checker | not started | October 30, 2026 |",
        "| Cross-dataset comparison for both classifiers | started | October 31, 2026 |",
        "| Limitations, final report and AI-use disclosure | started | November 1, 2026 |",
        "| Fresh-copy reproduction, repository publication and final review | not started | November 2, 2026 |", "",
        "### (b) Main comparison", "",
        "Variable: test dataset (UNSW-NB15 official test, CICIDS2017, TON_IoT). For each classifier, hold training rows, preprocessing, representation, fitted model and validation-selected threshold fixed. Use the same UNSW validation assignments for RF and XGBoost; each model selects its own threshold with the same <=5% validation-FPR rule. Report each model as a separate results-table row. No test-driven retuning is planned.", "",
        f"Primary metric: AUROC. Operational question: does source-to-external AUROC fall by a meaningful amount? Let drop = UNSW AUROC minus external AUROC. Count yes when drop >= {margin:.3f}, no meaningful decline when drop <= 0, and a smaller positive drop as inconclusive under this decision margin. Apply the rule separately to each external dataset and each classifier. The widest printed baseline AUROC interval has half-width {half_width:.4f}; the {margin:.3f} yes margin exceeds it. This rule is specified after observing the baseline, so it is a planning rule for remaining work, not a preregistered confirmatory hypothesis.", "",
        "| Classifier | UNSW test AUROC | CICIDS2017 AUROC | TON_IoT AUROC |", "|---|---:|---:|---:|",
        "| Random Forest | " + " | ".join(results[n]['AUROC']['printed'] for n in ['UNSW-NB15 official test','CICIDS2017','TON_IoT']) + " |", "| XGBoost | pending | pending | pending |", "",
        "### (c) Compute budget", "",
        f"Measured RF fit: {measured_rf:.4f} min; scoring and point metrics: {scoring:.2f} min; course checker: {checker:.2f} min. Budget one XGBoost fit at 10x RF fit ({10*measured_rf:.2f} min, a provisional multiplier, not a measurement), plus one XGBoost scoring/checker pass and one RF reproduction pass. Estimated remaining compute = 11 x RF fit + 2 x RF scoring time + 2 x checker time = {compute_estimate:.2f} min. Reserve twice that ({2*compute_estimate:.2f} min) for variation; this excludes downloads, repeated full audits and writing. Replace the XGBoost estimate with its measured runtime after the first run.", "",
        "### AI-use disclosure draft", "",
        "OpenAI Codex assisted with data-audit, feature-alignment, splitting and Random Forest pipeline code; tests; interpretation; and preparation of this results/plan draft. Course-checker output supplies the reported metrics. The author should review the work and format this disclosure according to the syllabus; the syllabus's exact citation requirements were not supplied.", ""]
    verification_path = ROOT / "results/reproduction/verification.json"
    if verification_path.exists():
        verification = json.loads(verification_path.read_text())
        if verification["status"] in {"passed", "completed_with_floating_point_differences"}:
            for i, line in enumerate(lines):
                if line.startswith("It rebuilds the audit,"):
                    lines[i] = "The complete command was executed successfully in an isolated source/output workspace using the existing locked environment and original raw inputs. Assignments, fitted tree arrays and all thresholded test predictions reproduced. Probabilities differed by at most " + str(verification["maximum_absolute_score_difference"]) + "; tiny score-tie changes slightly altered rank metrics, so the original saved scores remain the source of this document's numbers. See results/reproduction/verification.json. A fresh package installation, fresh Git clone and independent data downloads were not tested. Latest artifacts still need to be committed and pushed."
                if "| Fresh-copy reproduction, repository publication and final review | not started |" in line:
                    lines[i] = line.replace("| not started |", "| started (isolated rebuild verified) |")
    text = "\n".join(lines)
    (OUT / "results_and_finish_plan.md").write_text(text, encoding="utf-8")
    (OUT / "finish_plan.md").write_text(text[text.index("## 5. Plan to finish"):], encoding="utf-8")
    (OUT / "checker_metrics.json").write_text(json.dumps({"tests": results, "widest_AUROC_half_width_from_printed_intervals": half_width, "meaningful_drop": margin, "warnings": warnings}, indent=2) + "\n")
    # Self-contained PDF supplement, including the actual required figure.
    pages = []
    current = []
    for line in lines:
        if line.startswith("!["):
            continue
        # Plain PDF keeps all table numbers; wrap long table rows without clipping.
        wrapped = textwrap.wrap(line, width=108, replace_whitespace=False) or [""]
        for fragment in wrapped:
            if len(current) == 65:
                pages.append(current); current = []
            current.append(fragment)
    if current:
        pages.append(current)
    with PdfPages(OUT / "results_and_finish_plan.pdf") as pdf:
        for i, page in enumerate(pages):
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.text(.06, .96, "\n".join(page), va="top", family="monospace", fontsize=8, linespacing=1.35)
            fig.text(.5, .025, f"Project 11 | Results and finish plan | {i+1}", ha="center", fontsize=8)
            pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.imshow(plt.imread(BASE / "unsw_confusion_matrix.png")); ax.axis("off")
        fig.text(.08, .12, "UNSW-NB15 official test: 82,332 rows; 55.06% attacks.\nAt the validation-selected threshold: recall 20.84%, FPR 10.92%.", fontsize=10)
        pdf.savefig(fig); plt.close(fig)
    old = (BASE / "baseline_report.md").read_text(encoding="utf-8")
    old = re.sub(r"Point estimates below are preliminary\..*?Checker status: pending\.", "Course checker completed. Official metrics and 95% intervals are in checker_output.txt; the checkpoint-ready interpretation and plan are in ../checkpoint/results_and_finish_plan.md.", old)
    (BASE / "baseline_report.md").write_text(old, encoding="utf-8")
    artifact_checks_path = BASE / "artifact_checks.json"
    if artifact_checks_path.exists():
        artifact_checks = json.loads(artifact_checks_path.read_text())
        artifact_checks["course_checker"] = "completed; see checker_output.txt"
        artifact_checks_path.write_text(json.dumps(artifact_checks, indent=2) + "\n")
    print("Wrote checkpoint results/plan Markdown, PDF and checker metrics JSON.")


if __name__ == "__main__":
    main()
