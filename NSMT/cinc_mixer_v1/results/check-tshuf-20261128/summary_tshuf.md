# MIT-BIH order-only control: SpkMixer trained and tested with a random token order per sample

Rules: summarize_tshuf.py docstring. 3 new runs, commit `ab3aeba47`.

| arm | macro-F1 % | accuracy % | F1 N / S / V / F / Q % | energy µJ (syops) |
|---|--:|--:|---|--:|
| SpkMixer order-shuffled (new) | 90.33 | 98.05 | 98.96 / 79.54 / 95.19 / 79.51 / 98.44 | 36.89 |
| SpkMixer base | 91.99 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 37.06 |
| SpkMixer memoryless | 90.13 | 98.03 | 98.94 / 78.59 / 95.14 / 79.42 / 98.58 | 36.15 |
| SF-de PE off | 91.27 | 98.33 | 99.11 / 82.07 / 95.93 / 80.57 / 98.68 | 7.62 |
| SF-de PE on | 91.94 | 98.53 | 99.23 / 85.14 / 96.09 / 80.23 / 99.03 | 7.80 |

| kind | A | B | d %p (42 / 7 / 2026) | mean %p | verdict | energy A/B |
|---|---|---|---|--:|---|--:|
| S1 | SpkMixer order-shuffled (new) | SpkMixer base | -1.74 / -1.54 / -1.70 | -1.66 | **A worse** | 0.995 |
| S2 | SpkMixer order-shuffled (new) | SpkMixer memoryless | -0.02 / +0.23 / +0.37 | +0.19 | **no clear difference** | 1.020 |
| S3 | SpkMixer order-shuffled (new) | SF-de PE off | -1.29 / -0.52 / -1.03 | -0.95 | **A worse** | 4.841 |
| S4 | SpkMixer order-shuffled (new) | SF-de PE on | -1.61 / -1.92 / -1.32 | -1.62 | **A worse** | 4.729 |

Decomposition (macro-F1 %p, seed 42 / 7 / 2026): token order = base - shuffled +1.74 / +1.54 / +1.70 (mean +1.66); token interaction without order = shuffled - memoryless -0.02 / +0.23 / +0.37 (mean +0.19)

Evaluation-order spread (report): seed 2026: default 90.25, eval seed 1 90.23, seed 2 90.64, natural order 89.19 (spread 0.42 %p); seed 42: default 90.15, eval seed 1 89.97, seed 2 89.79, natural order 89.12 (spread 0.36 %p); seed 7: default 90.59, eval seed 1 89.97, seed 2 90.68, natural order 88.39 (spread 0.71 %p)
