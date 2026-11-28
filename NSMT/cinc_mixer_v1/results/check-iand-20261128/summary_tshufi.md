# MIT-BIH order-only control on the IAND base: SpkMixer IAND trained and tested with a random token order per sample

Rules: summarize_tshufi.py docstring. 3 new runs, commit `5a888c310`.

| arm | macro-F1 % | accuracy % | F1 N / S / V / F / Q % | energy µJ (syops) |
|---|--:|--:|---|--:|
| SpkMixer IAND order-shuffled (new) | 90.34 | 98.17 | 99.03 / 81.07 / 95.65 / 77.39 / 98.56 | 8.50 |
| SpkMixer IAND base | 92.21 | 98.59 | 99.27 / 86.42 / 96.31 / 80.23 / 98.83 | 9.00 |
| SpkMixer ADD base | 91.99 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 37.06 |
| SpkMixer ADD order-shuffled | 90.33 | 98.05 | 98.96 / 79.54 / 95.19 / 79.51 / 98.44 | 36.89 |
| SpkMixer ADD memoryless | 90.13 | 98.03 | 98.94 / 78.59 / 95.14 / 79.42 / 98.58 | 36.15 |
| SF-de PE off | 91.27 | 98.33 | 99.11 / 82.07 / 95.93 / 80.57 / 98.68 | 7.62 |
| SF-de PE on | 91.94 | 98.53 | 99.23 / 85.14 / 96.09 / 80.23 / 99.03 | 7.80 |

| kind | A | B | d %p (42 / 7 / 2026) | mean %p | verdict | energy A/B |
|---|---|---|---|--:|---|--:|
| T1 | SpkMixer IAND order-shuffled (new) | SpkMixer IAND base | -2.31 / -2.34 / -0.96 | -1.87 | **A worse** | 0.944 |
| T2 | order share IAND | order share ADD | +0.57 / +0.79 / -0.73 | +0.21 | **no clear difference** | — |
| T3 (report) | SpkMixer IAND order-shuffled (new) | SF-de PE off | -1.48 / -0.88 / -0.43 | -0.93 | **A worse** | 1.115 |
| T4 (report) | SpkMixer IAND order-shuffled (new) | SF-de PE on | -1.81 / -2.28 / -0.72 | -1.60 | **A worse** | 1.089 |
| T5 (report) | SpkMixer IAND order-shuffled (new) | SpkMixer ADD order-shuffled | -0.20 / -0.36 / +0.60 | +0.02 | **no clear difference** | 0.230 |
| T6 (report) | SpkMixer IAND order-shuffled (new) | SpkMixer ADD memoryless | -0.22 / -0.13 / +0.97 | +0.21 | **no clear difference** | 0.235 |

Order share (macro-F1 %p, seed 42 / 7 / 2026): IAND base - IAND shuffled +2.31 / +2.34 / +0.96 (mean +1.87); ADD base - ADD shuffled +1.74 / +1.54 / +1.70 (mean +1.66)

Evaluation-time token shuffle of the trained base (no training; drop in macro-F1 %p, seed 42 / 7 / 2026, mean of 3 permutation seeds): IAND mixer input 34.99 / 32.12 / 31.65 (mean 32.92), patch tokens 34.99 / 32.12 / 31.65 (mean 32.92); ADD mixer input 35.75 / 35.07 / 34.35 (mean 35.06)

Evaluation-order spread of the IAND shuffled model (report): seed 2026: default 90.85, eval seed 1 90.76, seed 2 91.08, natural order 90.14 (spread 0.33 %p); seed 42: default 89.95, eval seed 1 90.23, seed 2 90.49, natural order 89.30 (spread 0.54 %p); seed 7: default 90.23, eval seed 1 90.51, seed 2 90.77, natural order 89.59 (spread 0.55 %p)
