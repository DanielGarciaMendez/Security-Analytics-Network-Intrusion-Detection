# Section 2(c) — Pre-split boundary investigation

**Status: PREVIEW ONLY.** No raw records were modified, no rows removed, and no final train/validation assignments made.
The values below come from locally loaded, feature-aligned CSVs; they are not publisher statistics.

## Shared feature mapping (candidate v0.1)

| Canonical input | UNSW-NB15 | CICIDS2017 | TON_IoT | Unit conversion |
|---|---|---|---|---|
| `duration_seconds` | `dur` | `Flow Duration` | `duration` | CICIDS / 1,000,000; others unchanged |
| `forward_packets` | `spkts` | `Total Fwd Packets` | `src_pkts` | None |
| `backward_packets` | `dpkts` | `Total Backward Packets` | `dst_pkts` | None |

The originator/forward packet definitions and flow timeouts may differ among collectors. Byte measures were withheld pending unit/definition verification. The candidate map must be approved before final splitting.

## Source partitions and preliminary overlap impact

Original UNSW training source: **175,341 rows** (benign=56,000, attack=119,341).
Invalid aligned UNSW training feature rows: **0** (missing, infinite, or negative).

| Planned test set | Original test rows | Benign | Attack | Invalid model inputs | Test rows matching a train pattern | Training rows matching this test set |
|---|---:|---:|---:|---:|---:|---:|
| UNSW-NB15 official test | 82,332 | 37,000 | 45,332 | 0 | 39,765 | 89,652 |
| CICIDS2017 | 2,830,743 | 2,273,097 | 557,646 | 115 | 217,912 | 84,857 |
| TON_IoT | 211,043 | 50,000 | 161,043 | 0 | 56,712 | 80,130 |

**Definition:** A match requires exact numeric equality of *all three aligned model inputs* (without rounding or including labels). Numbers are counted as float64 after duration-unit conversion; negative zero is normalized to zero. Invalid model-input rows are not compared.
Training-row counts for individual test sets **overlap**, so do not add them.

## Hypothetical strict three-feature exclusion from UNSW training

Distinct train input patterns that occur in at least one test set: **2,056**.
Training rows affected by >=1 test set: **90,275** (51.49% of the original training source).
Rows remaining *if* all matching training rows and invalid feature rows were excluded: **85,066** (benign=40,254, attack=44,812).

**This is not a deletion instruction.** With only three numerical features, unrelated flows can have the same measurements. Strict exclusion can remove representative behavior and bias the experiment. Consider the loss of each class before approving this rule.

**CAUTION:** The strict three-feature rule would exclude over half of the original training records; do not apply it blindly.

## Primary-dataset near-duplicate screening boundary

UNSW coarse-group screening key: same `proto`, `service`, `state`, `spkts`, `dpkts`; duration rounded to 0.01 seconds, and total `sbytes + dbytes` rounded to nearest 100. This uses the original files and is deliberately only a candidate screen.
- Distinct UNSW coarse groups found in both official partitions: **6,032**.
- Training records belonging to those common coarse groups: **127,203** (72.55%).
- Coarse-group overlap does not establish the same real event, and this count is not a recommended removal count.
- This rule is wider than the audit's near-duplicate candidate rule because it does not require two different full feature vectors within each coarse group.

## Before finalizing Section 2(c)

1. Review the shared feature mapping and the potential strict-exclusion impact; document a defensible decision.
2. Choose primary near-duplicate grouping/protection rule without treating every coarse match as the same event.
3. Create group-aware UNSW training/validation assignments with a fixed seed and class counts.
4. Preserve separate UNSW official, CICIDS2017 and TON_IoT test rows, or document any excluded test rows.
5. Produce `split.csv`, count exclusions and which boundary they protect; then run Section 2(d) final leakage checks.

**Do not call this output the final split table.** It is the measured evidence needed to choose the split rule.
