# Random Forest checkpoint baseline

Training: 68,048 rows; validation: 17,018 rows. Seed: 42.
Fit time: 2.02 seconds on {'platform': 'Windows-10-10.0.19045-SP0', 'processor': 'AMD64 Family 23 Model 113 Stepping 0, AuthenticAMD', 'logical_cpus': 12}. No refit after validation.

Features: duration_seconds, forward_packets, backward_packets. No scaling or imputation. 200 trees, min_samples_leaf=2; other settings are recorded in run_metadata.json.

Positive class: attack (1). Threshold: **0.9426192528735632**, selected only on validation using a 5% empirical false-positive cap. Validation FPR: 4.9895%; recall: 55.3175%.

Course checker completed. Official metrics and 95% intervals are in checker_output.txt; the checkpoint-ready interpretation and plan are in ../checkpoint/results_and_finish_plan.md.

| Classifier | UNSW-NB15 official test AUROC | CICIDS2017 AUROC | TON_IoT AUROC |
|---|---:|---:|---:|
| Random Forest | 0.783270 | 0.504833 | 0.516922 |
| XGBoost (pending) | — | — | — |

![UNSW confusion matrix](unsw_confusion_matrix.png)

Figure: UNSW-NB15 official test, 82,332 rows, 55.06% attacks. At the validation-selected threshold, test recall is 20.84% and test FPR is 10.92%; the validation FPR cap is not a guarantee for other data.

Limitations: three-feature representation, collector differences, strict training-pattern exclusions, and unavailable UNSW host/session identifiers. Row-bootstrap intervals do not establish independence among repeated or related flows. Test results were not used to tune this baseline.

