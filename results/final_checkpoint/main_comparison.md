# Section 1. Main comparison

## 1(a) What was run

Does intrusion-detection performance deteriorate when a model trained on
UNSW-NB15 is transferred to other network datasets? The primary metric is
AUROC, and the direction of interest is lower AUROC on external test data.
The test dataset is the experimental variable. The three values are
`UNSW-NB15 official test` (baseline), `CICIDS2017`, and `TON_IoT`. The
classifier column is `Random Forest`.

The same fitted Random Forest, UNSW-NB15 training and validation rows,
preprocessing, decision threshold, and three inputs were used for all three
tests. The inputs are duration in seconds, forward packet count, and backward
packet count. CICIDS2017 durations were converted from microseconds to seconds.
The aligned features were used without scaling or imputation. Benign is label
0 and attack is label 1.

UNSW-NB15 supplied 68,048 training rows (32,197 benign and 35,851 attacks) and
17,018 validation rows (8,057 benign and 8,961 attacks). The saved split search
and Random Forest both used seed 42. The comparison includes every retained
test flow across all three datasets, with no random test subsampling. For
intervals, the final checker sorts by row ID and seeds each cell's bootstrap
using the CRC32 hashes of its value and column. It uses 1,000 resamples for
UNSW-NB15 and 200 each for CICIDS2017 and TON_IoT. Seed 42 for the separate
error sample has no role in the comparison metrics.

The midsemester UNSW-trained Random Forest experiment carries forward. The
final checkpoint reformats its published scores into the required dataset
rows and classifier column, adds the error sample, and uses the final
checker's deterministic bootstrap and baseline comparisons. The fitted
model, threshold, split assignments, and score values are those already saved.
This comparison covers the completed UNSW-trained Random Forest cells. The
midsemester finish plan also described additional training sources and
classifiers, which have no scored cells in this run.

## 1(b) Threshold

The saved Random Forest threshold is `0.9426192528735632`. It is the lowest
observed score threshold on UNSW-NB15 validation data with empirical
false-positive rate at most 5%, keeping score ties together. The observed
validation FPR was 4.9895%. The same threshold was applied to all three test
datasets, with scores at or above it classified as attack. Test labels were
used for evaluation, and threshold selection used validation data only.
AUROC measures ranking across operating points.
[Run metadata](../baseline/run_metadata.json)

## 1(c) Leakage

**Same training set.** All three evaluations use the same UNSW-NB15 fit.
The recorded split checks were:

| Check | Count |
|---|---:|
| Training rows exactly matching a validation feature triple | 0 |
| Training rows exactly matching any retained test feature triple | 0 |
| Operational near-groups crossing training and validation | 0 |
| Operational near-groups crossing development and official UNSW-NB15 test | 0 |
| Connected groups crossing training and validation | 0 |

The split process excluded 90,275 UNSW-NB15 training-source rows with exact
aligned-input matches to test flows (15,746 benign and 74,529 attacks) and
zero additional rows through grouping. It excluded 115 benign CICIDS2017
rows with invalid aligned inputs. Of 3,299,459 original rows, 3,209,069 were
retained and 90,390 excluded.

Exact overlap is defined on the three aligned features. The near-group rule
applies to UNSW-NB15 and groups equal values of the other 39 network features
with duration rounded to 0.01 seconds and total bytes rounded to the nearest
100. Within-partition duplicates and overlap between test sets were allowed.
Host/session identifiers and time-based independence were unavailable for the
prepared UNSW-NB15 files. These checks address specific overlap mechanisms,
and excluding common feature triples changes the UNSW training distribution.
[Final split report](../splits/section_2c_final.md) ·
[Recorded leakage checks](../splits/leakage_checks.json)

The highest metric point estimate was 0.8627 (TON_IoT precision). The checker
reported no metric reaching 0.99 and issued no leakage warning.

## 1(d) Results

| Test dataset | Random Forest AUROC [95% CI] |
|---|---|
| UNSW-NB15 official test | 0.7833 [0.7797, 0.7866] |
| CICIDS2017 | 0.5048 [0.5040, 0.5057] |
| TON_IoT | 0.5169 [0.5144, 0.5195] |

Cells with AP at or below random-scorer AP or AUROC at or below 0.5

None.

| Test dataset | Test rows | Benign rows | Attack rows (%) | AP | Random-scorer AP |
|---|---:|---:|---:|---:|---:|
| UNSW-NB15 official test | 82,332 | 37,000 | 45,332 (55.06%) | 0.7530 | 0.5506 |
| CICIDS2017 | 2,830,628 | 2,272,982 | 557,646 (19.70%) | 0.2829 | 0.1970 |
| TON_IoT | 211,043 | 50,000 | 161,043 (76.31%) | 0.7943 | 0.7631 |

Random-scorer AP equals the positive fraction. All three AP values exceed
their dataset-specific reference, and all three AUROC values exceed 0.5.

## 1(e) Changes and verdicts

The checker computes the difference as external AUROC minus baseline AUROC.
With direction `lower`, it reverses the sign to report the change toward
lower AUROC. The checker computes these differences before rounding the
displayed AUROC values.

| Comparison | AUROC difference | Change toward lower AUROC | Verdict |
|---|---:|---:|---|
| CICIDS2017 vs. UNSW-NB15 official test | -0.2784 | +0.2784 | Pending verification |
| TON_IoT vs. UNSW-NB15 official test | -0.2663 | +0.2663 | Pending verification |

The available midsemester report (`MidsemseterCheckPoint_DanielGarciaMendez.pdf`
and the parallel October 29 copy) names AUROC in Part A(e). Its Part C(b),
page 9, describes the 0.050 decline margin as a planning rule chosen after
seeing the initial baseline. The repository's finish plan proposes a Yes
difference of 0.050 or more and a No difference of 0 or less. The proposal,
midsemester copies, and repository history did not establish those as an
original prespecified pair. Both verdicts therefore remain pending. Once
verified, the handout assigns Yes at or above the Yes difference, No at or
below the No difference, and Undecided between them or for an unavailable
change. [Repository finish plan](../checkpoint/finish_plan.md)

The source test AUROC of 0.7833 shows substantially better ranking than random
scoring on UNSW-NB15. On CICIDS2017, ranking fell to 0.5048, close to random.
On TON_IoT, it fell to 0.5169, still close to random despite being higher
than CICIDS2017. Both external 95% row-bootstrap intervals lie well below
the UNSW interval, and both lie above 0.5 under the checker's resampling
scheme. These intervals describe per-cell uncertainty when resampling rows
from the observed datasets. The checker supplies point differences and
per-cell intervals, so the intervals alone do not provide a confidence
interval or significance test for the drop. Practical importance requires
the verified decision rule. The results show poorer transfer for this fixed,
three-feature Random Forest on these two datasets. They do not establish the
cause of the drops or performance on other datasets, feature sets, or models.

## Checker verification

The final checker exited with status 0 using `--metric auroc`,
`--reference "UNSW-NB15 official test"`, and `--direction lower`. It
validated 10 false positives and 10 false negatives from the existing error
sample and printed no warnings. The before and after SHA-256 hashes matched
for `split.csv.gz`, `scores.csv.gz`, and `errors.csv`.
[Complete checker output](final_checker_output.txt) · [Error analysis](error_analysis.md)
