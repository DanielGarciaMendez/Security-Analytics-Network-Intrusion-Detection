# Section 2(b) — Dataset Audit

**Primary dataset:** UNSW-NB15. The baseline will train on its official training CSV.

**Scope of this table:** All ORIGINAL loaded files were audited BEFORE cleaning: 2 UNSW CSVs, 8 CICIDS2017 CSVs and 1 TON_IoT network CSV. UNSW counts combine its official training and testing files, with file-level counts retained separately.

| Audit check | UNSW-NB15 (primary) | CICIDS2017 | TON_IoT |
|---|---:|---:|---:|
| Rows | 257,673 | 2,830,743 | 211,043 |
| Columns | 45 | 79 | 44 |
| Benign count (%) | 93,000 (36.09%) | 2,273,097 (80.30%) | 50,000 (23.69%) |
| Malicious count (%) | 164,673 (63.91%) | 557,646 (19.70%) | 161,043 (76.31%) |
| Missing target values | 0 | 0 | 0 |
| Missing key features | 0 [A] | — | — |
| Exact duplicates | 94,928 [B] | 308,381 [B] | 20,569 [B] |
| Near-duplicates | 14,431 candidate rows / 20,000 sampled [C] | — | — |
| Repeated entities | Unavailable — no host IDs [D] | — | — |
| Time range | Unavailable — no timestamp [E] | Not required [F] | Not required [F] |

## UNSW-NB15 official partition detail

The main audit table combines the two source files; the **baseline trains on the official training file only**.

| Source file | Rows | Benign | Attack | Duplicate occurrences inside this file (excluding id) |
|---|---:|---:|---:|---:|
| `UNSW_NB15_training-set.csv` | 175,341 | 56,000 | 119,341 | 67,601 |
| `UNSW_NB15_testing-set.csv` | 82,332 | 37,000 | 45,332 | 26,387 |

## Definitions and limitations

**[A] Key features.** UNSW missing-value count is the sum across `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes` and `rate`. Missing counts were calculated on the loaded CSVs. 

**[B] Exact duplicates.** Duplicate occurrences **after the first instance**, counted across all loaded CSVs *within each dataset*, using two independent 64-bit row fingerprints. Equality fields: UNSW = every loaded column **except `id`** (includes `attack_cat` and `label`); CICIDS2017 = **all 79 columns** (includes `Label`); TON_IoT = **all 44 columns** (includes `label` and `type`). Fingerprint collisions are theoretically possible. Counts may differ from older reports listing only *within-file* duplicates. These counts are NOT the train/test leakage check.

**[C] Near-duplicates (primary dataset only).** Sample of **20,000 rows** from the UNSW official **training** file, random seed **42**. Checking for near-duplicates include: Same proto, service, state, spkts and dpkts; duration rounded to 0.01 seconds; total bytes rounded to nearest 100; at least two distinct full 42-feature network patterns within the group. Reported count is the number of **sample rows participating in candidate groups**, not the number of distinct pairs and not confirmation of the same event.

**[D] Repeated entities.** There are not any "repeated entities" for our main dataset UNSW CSV's, as there is a unique ID for every column.

**[E] Time range.** Unavailable for UNSW because neither loaded, prepared CSV contains a per-record timestamp column.

**[F] Other datasets.** A per-record time range is not required for CICIDS2017 and TON_IoT because this experiment uses a **dataset-boundary**, not time-boundary, split.


