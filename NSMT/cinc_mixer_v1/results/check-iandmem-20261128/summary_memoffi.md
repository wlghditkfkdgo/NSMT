# MIT-BIH memoryless control on the IAND base (SpkMixer IAND, token-axis mixer neurons reset after each token)

Rules: summarize_memoffi.py docstring. 3 new runs, commit `466da4420`.

| arm | macro-F1 % | accuracy % | F1 N / S / V / F / Q % | energy µJ (syops) |
|---|--:|--:|---|--:|
| SpkMixer IAND memoryless (new) | 91.08 | 98.22 | 99.05 / 81.20 / 95.36 / 81.05 / 98.75 | 8.03 |
| SpkMixer IAND base | 92.21 | 98.59 | 99.27 / 86.42 / 96.31 / 80.23 / 98.83 | 9.00 |
| SpkMixer IAND order-shuffled | 90.34 | 98.17 | 99.03 / 81.07 / 95.65 / 77.39 / 98.56 | 8.50 |
| SpkMixer ADD memoryless | 90.13 | 98.03 | 98.94 / 78.59 / 95.14 / 79.42 / 98.58 | 36.15 |
| SF-de PE off | 91.27 | 98.33 | 99.11 / 82.07 / 95.93 / 80.57 / 98.68 | 7.62 |
| SF-de PE on | 91.94 | 98.53 | 99.23 / 85.14 / 96.09 / 80.23 / 99.03 | 7.80 |

| kind | A | B | d %p (42 / 7 / 2026) | mean %p | verdict | energy A/B |
|---|---|---|---|--:|---|--:|
| U1 | SpkMixer IAND memoryless (new) | SpkMixer IAND base | -1.27 / -1.16 / -0.96 | -1.13 | **A worse** | 0.892 |
| U2 | SpkMixer IAND order-shuffled | SpkMixer IAND memoryless (new) | -1.04 / -1.17 / -0.01 | -0.74 | **A worse** | 1.058 |
| U3 (report) | SpkMixer IAND memoryless (new) | SpkMixer ADD memoryless | +0.82 / +1.05 / +0.97 | +0.95 | **A better** | 0.222 |
| U4 (report) | SpkMixer IAND memoryless (new) | SF-de PE off | -0.44 / +0.29 / -0.42 | -0.19 | **no clear difference** | 1.054 |
| U5 (report) | SpkMixer IAND memoryless (new) | SF-de PE on | -0.76 / -1.10 / -0.72 | -0.86 | **A worse** | 1.030 |

Decomposition (macro-F1 %p, seed 42 / 7 / 2026): base - memoryless +1.27 / +1.16 / +0.96 (mean +1.13) = order share (base - shuffled) +2.31 / +2.34 / +0.96 (mean +1.87) + interaction share (shuffled - memoryless) -1.04 / -1.17 / -0.01 (mean -0.74); order fraction of the mean drop 1.655

Check: evaluation-time token shuffle of the memoryless IAND model, drop in macro-F1 %p (mixer input / patch tokens): seed 2026 +0.0000 / +0.0000; seed 42 +0.0000 / +0.0000; seed 7 +0.0000 / +0.0000 (|drop| < 0.05 required)
