## 5. Plan to finish

### (a) Required elements

Proposed dates based on the work remaining on October 29, 2026; these are planning targets, not an assumed course deadline. Scope follows the available proposal (Random Forest and XGBoost); reconcile any additional instructor-assigned requirements separately.

| Required element | Status | Completion/target date |
|---|---|---|
| Three datasets, binary labels and shared feature mapping | done | October 28, 2026 |
| Audit, exclusions, group-aware splits and leakage checks | done | October 28, 2026 |
| Recent dataset-paper citations | done (user-confirmed) | October 29, 2026 or earlier |
| Random Forest training, validation threshold and three test sets | done | October 28, 2026 |
| Course checker, intervals, figure and results interpretation | done | October 29, 2026 |
| XGBoost baseline on the same features and assignments | not started | October 30, 2026 |
| XGBoost validation-only threshold, test scores and checker | not started | October 30, 2026 |
| Cross-dataset comparison for both classifiers | started | October 31, 2026 |
| Limitations, final report and AI-use disclosure | started | November 1, 2026 |
| Fresh-copy reproduction, repository publication and final review | started (isolated rebuild verified) | November 2, 2026 |

### (b) Main comparison

Variable: test dataset (UNSW-NB15 official test, CICIDS2017, TON_IoT). For each classifier, hold training rows, preprocessing, representation, fitted model and validation-selected threshold fixed. Use the same UNSW validation assignments for RF and XGBoost; each model selects its own threshold with the same <=5% validation-FPR rule. Report each model as a separate results-table row. No test-driven retuning is planned.

Primary metric: AUROC. Operational question: does source-to-external AUROC fall by a meaningful amount? Let drop = UNSW AUROC minus external AUROC. Count yes when drop >= 0.050, no meaningful decline when drop <= 0, and a smaller positive drop as inconclusive under this decision margin. Apply the rule separately to each external dataset and each classifier. The widest printed baseline AUROC interval has half-width 0.0034; the 0.050 yes margin exceeds it. This rule is specified after observing the baseline, so it is a planning rule for remaining work, not a preregistered confirmatory hypothesis.

| Classifier | UNSW test AUROC | CICIDS2017 AUROC | TON_IoT AUROC |
|---|---:|---:|---:|
| Random Forest | 0.7833 [0.7799, 0.7867] | 0.5048 [0.5039, 0.5060] | 0.5169 [0.5139, 0.5191] |
| XGBoost | pending | pending | pending |

### (c) Compute budget

Measured RF fit: 0.0337 min; scoring and point metrics: 1.14 min; course checker: 5.73 min. Budget one XGBoost fit at 10x RF fit (0.34 min, a provisional multiplier, not a measurement), plus one XGBoost scoring/checker pass and one RF reproduction pass. Estimated remaining compute = 11 x RF fit + 2 x RF scoring time + 2 x checker time = 14.12 min. Reserve twice that (28.23 min) for variation; this excludes downloads, repeated full audits and writing. Replace the XGBoost estimate with its measured runtime after the first run.

### AI-use disclosure draft

OpenAI Codex assisted with data-audit, feature-alignment, splitting and Random Forest pipeline code; tests; interpretation; and preparation of this results/plan draft. Course-checker output supplies the reported metrics. The author should review the work and format this disclosure according to the syllabus; the syllabus's exact citation requirements were not supplied.
