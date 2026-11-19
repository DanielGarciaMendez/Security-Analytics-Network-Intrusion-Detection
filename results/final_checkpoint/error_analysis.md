# Section 4. Error analysis

## A. Sampling methodology

The selected cell is **UNSW-NB15 official test / Random Forest**. It contains 37,000 benign flows (0) and 45,332 attack flows (1). A score at or above the saved threshold `0.9426192528735632` predicts attack.

The cell has **4,039 false positives** and **35,887 false negatives**. I sorted each error pool by row ID, then used one Python `random.Random(42)` instance to sample without replacement, drawing false positives first. The draw yielded 10 false positives and 10 false negatives (20 total).

I matched each sampled ID to the split and aligned features, then used the zero-based source row index encoded in the ID (excluding the header) to locate its raw flow. Raw labels, duration, and packet counts matched the aligned records.

## B. Error category summary

First check for zero backward packets and separate single-packet from multi-packet flows. For bidirectional flows, compare packet counts and group ties with forward-heavy flows. These rules are mutually exclusive and use the same criteria for both error types.

| Category | One-sentence rule | False positives | False negatives | Example row ID |
|---|---|---:|---:|---|
| Single-packet with no reply | Exactly one forward packet and zero backward packets are recorded. | 3 | 0 | `unsw_test:UNSW_NB15_testing-set:41861` |
| Multiple packets with no reply | At least two forward packets and zero backward packets are recorded. | 0 | 7 | `unsw_test:UNSW_NB15_testing-set:16556` |
| Reverse-heavy bidirectional | Both directions have packets and the backward count exceeds the forward count. | 0 | 1 | `unsw_test:UNSW_NB15_testing-set:971` |
| Forward-heavy or balanced bidirectional | Both directions have packets and the forward count is at least the backward count. | 7 | 2 | `unsw_test:UNSW_NB15_testing-set:74898` |

## C. Individual sampled errors

Every row uses the threshold in Section A. Opposite-label matches count scored rows in this cell with exactly the same duration and two packet counts. Protocol, service, state, bytes, and attack family come from the original dataset. A dash in the service column is the source value. Packet and byte counts are shown as forward / backward.

| Row ID | Actual | Predicted | Score | Duration (s) | Packets | Protocol / service / state | Bytes | Source attack family | Opposite-label matches | Category |
|---|---:|---:|---:|---:|---|---|---|---|---:|---|
| `unsw_test:UNSW_NB15_testing-set:74898` | 0 | 1 | 0.9876806984201716 | 0.621471 | 10 / 8 | tcp / - / FIN | 2516 / 354 | Normal | 0 | Forward-heavy or balanced bidirectional |
| `unsw_test:UNSW_NB15_testing-set:41861` | 0 | 1 | 1.0 | 0.0 | 1 / 0 | arp / - / INT | 46 / 0 | Normal | 23 | Single-packet with no reply |
| `unsw_test:UNSW_NB15_testing-set:38859` | 0 | 1 | 0.9990625 | 1.651216 | 24 / 20 | tcp / - / FIN | 2158 / 1918 | Normal | 0 | Forward-heavy or balanced bidirectional |
| `unsw_test:UNSW_NB15_testing-set:77139` | 0 | 1 | 0.9468564330904202 | 0.400264 | 10 / 6 | tcp / - / FIN | 534 / 268 | Normal | 0 | Forward-heavy or balanced bidirectional |
| `unsw_test:UNSW_NB15_testing-set:67684` | 0 | 1 | 0.9538195873090604 | 1.170531 | 10 / 8 | tcp / - / FIN | 508 / 354 | Normal | 0 | Forward-heavy or balanced bidirectional |
| `unsw_test:UNSW_NB15_testing-set:67209` | 0 | 1 | 0.9841388888888889 | 4.668997 | 16 / 12 | tcp / - / FIN | 1860 / 642 | Normal | 0 | Forward-heavy or balanced bidirectional |
| `unsw_test:UNSW_NB15_testing-set:66815` | 0 | 1 | 0.9590281089172019 | 0.773026 | 10 / 6 | tcp / - / FIN | 578 / 268 | Normal | 0 | Forward-heavy or balanced bidirectional |
| `unsw_test:UNSW_NB15_testing-set:42339` | 0 | 1 | 1.0 | 0.0 | 1 / 0 | arp / - / INT | 46 / 0 | Normal | 23 | Single-packet with no reply |
| `unsw_test:UNSW_NB15_testing-set:77060` | 0 | 1 | 1.0 | 0.0 | 1 / 0 | arp / - / INT | 46 / 0 | Normal | 23 | Single-packet with no reply |
| `unsw_test:UNSW_NB15_testing-set:41730` | 0 | 1 | 0.9923620709691912 | 1.281334 | 10 / 8 | tcp / - / FIN | 756 / 354 | Normal | 0 | Forward-heavy or balanced bidirectional |
| `unsw_test:UNSW_NB15_testing-set:971` | 1 | 0 | 0.913120164410058 | 3.674717 | 114 / 656 | tcp / pop3 / FIN | 4986 / 856582 | Exploits | 0 | Reverse-heavy bidirectional |
| `unsw_test:UNSW_NB15_testing-set:16556` | 1 | 0 | 0.7810030247522117 | 7e-06 | 2 / 0 | udp / dns / INT | 114 / 0 | Generic | 219 | Multiple packets with no reply |
| `unsw_test:UNSW_NB15_testing-set:58980` | 1 | 0 | 0.7810030247522117 | 3.5e-05 | 2 / 0 | unas / - / INT | 200 / 0 | Fuzzers | 6 | Multiple packets with no reply |
| `unsw_test:UNSW_NB15_testing-set:13104` | 1 | 0 | 0.7810030247522117 | 9e-06 | 2 / 0 | udp / dns / INT | 114 / 0 | Generic | 1,050 | Multiple packets with no reply |
| `unsw_test:UNSW_NB15_testing-set:12979` | 1 | 0 | 0.7810030247522117 | 1.4e-05 | 2 / 0 | udp / dns / INT | 114 / 0 | Generic | 8 | Multiple packets with no reply |
| `unsw_test:UNSW_NB15_testing-set:16964` | 1 | 0 | 0.7810030247522117 | 8e-06 | 2 / 0 | udp / dns / INT | 114 / 0 | Generic | 462 | Multiple packets with no reply |
| `unsw_test:UNSW_NB15_testing-set:3986` | 1 | 0 | 0.7927541109388124 | 0.60885 | 12 / 12 | tcp / ftp / FIN | 574 / 682 | Fuzzers | 0 | Forward-heavy or balanced bidirectional |
| `unsw_test:UNSW_NB15_testing-set:44536` | 1 | 0 | 0.7810030247522117 | 1.3e-05 | 2 / 0 | udp / - / INT | 968 / 0 | Generic | 8 | Multiple packets with no reply |
| `unsw_test:UNSW_NB15_testing-set:65589` | 1 | 0 | 0.2909408972620016 | 0.69693 | 10 / 6 | tcp / - / FIN | 590 / 268 | Fuzzers | 1 | Forward-heavy or balanced bidirectional |
| `unsw_test:UNSW_NB15_testing-set:12755` | 1 | 0 | 0.7810030247522117 | 9e-06 | 2 / 0 | udp / dns / INT | 114 / 0 | Generic | 1,050 | Multiple packets with no reply |

## D. Interpretation

Three sampled false positives were benign ARP flows with zero recorded duration and a 1 / 0 packet count. They scored 1.0. Seven sampled false negatives had two forward packets and zero replies, with durations from 7 to 35 microseconds. They spanned Generic and Fuzzers labels and all scored 0.7810030247522117, below the cutoff.

Across the sample, 11 of 20 flows shared all three model inputs with at least one scored flow of the opposite label. The Random Forest receives identical inputs for each such pair, so its three-feature representation cannot distinguish those examples.

The bidirectional sample also shows why packet direction alone offers an incomplete explanation. Seven benign TCP FIN flows with forward packet majorities scored above the threshold, while the bidirectional attack examples included balanced, forward-heavy, and reverse-heavy flows below it. The reverse-heavy POP3 flow had 114 forward and 656 backward packets and scored about 0.9131. Its packet imbalance and source family describe the flow, but the sample provides no evidence about which model decisions produced that particular score.

Protocol, service, state, bytes, and attack family provide context only. The model used duration and the two packet counts. There is no packet-level evidence here to establish a root cause. The small sample deliberately balances error types within one test cell, so the category counts describe these sampled errors and do not estimate prevalence across the full test set.
