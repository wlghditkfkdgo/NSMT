# MIT-BIH: SNN ECG models of the NSMT paper Table 5 and our models (no training)

Sources and caveats: mitbih_lit.py docstring.

| model | Acc % | macro P % | macro R % | macro F1 % | energy | source |
|---|--:|--:|--:|--:|--:|---|
| Y. Xing et al. [44] | 98.26 | — | — | 89.09 | 346.33 µJ | NSMT Table 5 (from the cited paper) |
| A. Rana and K. Kim [30] | 93.80 | — | — | 86.40 | 278 (mW) | NSMT Table 5 (from the cited paper) |
| M. Scrugli et al. [33] | 98.40 | — | — | — | 50.98 µJ | NSMT Table 5 (from the cited paper) |
| K. Buettner and A. George [5] | 97.80 | — | — | 87.90 | 34.5 µJ | NSMT Table 5 (from the cited paper) |
| B. Reddy et al. [32] | 85.94 | — | — | — | — | NSMT Table 5 (from the cited paper) |
| NSMT (paper) | 98.45 | — | — | 91.91 | 19.05 µJ | NSMT Table 5 (from the cited paper) |
| SpkMixer IAND (base, lr 0.01) | 98.59 | 93.41 | 91.10 | 92.21 ± 0.31 | 9.00 µJ (syops) | logged, seeds 42 / 7 / 2026 |
| SpkMixer ADD (lr 0.02) | 98.56 | 92.89 | 91.14 | 91.99 ± 0.10 | 37.06 µJ (syops) | logged, seeds 42 / 7 / 2026 |
| NSMT rerun (same split / seeds) | 98.41 | 92.46 | 90.76 | 91.58 ± 0.05 | 19.06 µJ (syops) | logged, seeds 42 / 7 / 2026 |
| SF-de PE on (internal) | 98.53 | 93.19 | 90.78 | 91.94 ± 0.40 | 7.80 µJ (syops) | logged, seeds 42 / 7 / 2026 |

| SpkMixer IAND minus | Acc %p | F1 %p |
|---|--:|--:|
| Y. Xing et al. [44] | +0.33 | +3.12 |
| A. Rana and K. Kim [30] | +4.79 | +5.81 |
| M. Scrugli et al. [33] | +0.19 | — |
| K. Buettner and A. George [5] | +0.79 | +4.31 |
| B. Reddy et al. [32] | +12.65 | — |
| NSMT (paper) | +0.14 | +0.30 |

Paired, same split and seeds: SpkMixer IAND vs NSMT rerun -- f1 +0.62 / +0.99 / +0.28 (mean +0.63 %p, A better); overall_acc +0.21 / +0.21 / +0.13 (mean +0.18 %p, no clear difference); precision +1.12 / +1.20 / +0.53 (mean +0.95 %p, A better); sensitivity +0.18 / +0.81 / +0.05 (mean +0.35 %p, no clear difference); energy 0.472x
