# Order / accumulation check — flif-v1-oc (sha 4263abe29)

Pre-registered: docs/MODI_PROJECT_LOG.md 2026-11-26. Values in % (SC accuracy, others macro-F1), mean ± std over seeds (SC, MIT-BIH) or folds (Sleep-EDF); differences in %p, paired.

## Scores

| level | condition | metric | accuracy | permuted test (report only) |
|---|---|---|---|---|
| SLEEPCTX | gru_orig_L16 | 78.22 ± 3.61 | 84.33 ± 3.28 | 73.07 ± 2.54 |
| SLEEPCTX | gru_shuf_L16 | 77.07 ± 2.91 | 83.40 ± 3.51 | — |
| SLEEPCTX | tf_orig_L16 | 77.37 ± 4.09 | 83.34 ± 4.83 | 76.28 ± 4.05 |
| SLEEPCTX | tf_shuf_L16 | 76.78 ± 4.90 | 82.44 ± 5.32 | — |
| SLEEPCTX | gru_orig_L4 | 78.04 ± 4.05 | 83.71 ± 4.40 | 77.34 ± 3.95 |
| SLEEPCTX | gru_orig_L1 | 75.21 ± 2.28 | 82.70 ± 2.73 | 75.21 ± 2.28 |
| SLEEPIN | gru_orig | 73.94 ± 4.50 | 82.04 ± 4.17 | 71.49 ± 4.97 |
| SLEEPIN | gru_shuf | 74.55 ± 3.18 | 82.39 ± 3.43 | — |
| SLEEPIN | tf_orig | 73.09 ± 4.27 | 80.30 ± 4.12 | 72.92 ± 4.28 |
| SLEEPIN | tf_shuf | 73.65 ± 4.74 | 80.94 ± 4.68 | — |
| SC | gru_orig | 93.40 ± 0.07 | 93.40 ± 0.07 | 32.93 ± 1.36 |
| SC | gru_shuf | 91.34 ± 0.15 | 91.34 ± 0.15 | — |
| SC | tf_orig | 93.42 ± 0.23 | 93.42 ± 0.23 | 91.21 ± 0.28 |
| SC | tf_shuf | 93.19 ± 0.17 | 93.19 ± 0.17 | — |
| SC | gru_orig_k5 | 60.77 ± 0.25 | 60.77 ± 0.25 | 38.37 ± 0.41 |
| SC | gru_orig_k10 | 79.35 ± 0.38 | 79.35 ± 0.38 | 42.18 ± 0.34 |
| MITBIH | gru_orig | 91.72 ± 0.31 | 98.46 ± 0.05 | 25.95 ± 1.26 |
| MITBIH | gru_shuf | 89.08 ± 0.05 | 97.88 ± 0.02 | — |
| MITBIH | tf_orig | 91.48 ± 0.38 | 98.44 ± 0.07 | 38.91 ± 2.56 |
| MITBIH | tf_shuf | 89.85 ± 0.40 | 98.03 ± 0.06 | — |

## (B) order: orig − shuf

| level | model | d per pair | mean d | positive |
|---|---|---|---|---|
| SC | gru | +2.35 / +1.91 / +1.93 | +2.06 | 3/3 |
| SC | tf | +0.15 / +0.17 / +0.36 | +0.23 | 3/3 |
| MITBIH | gru | +3.01 / +2.73 / +2.20 | +2.65 | 3/3 |
| MITBIH | tf | +1.39 / +1.71 / +1.79 | +1.63 | 3/3 |
| SLEEPIN | gru | +0.17 / -2.05 / -4.20 / +2.53 / +0.52 | -0.60 | 3/5 |
| SLEEPIN | tf | -0.52 / -3.30 / -0.41 / +0.28 / +1.13 | -0.57 | 2/5 |
| SLEEPCTX | gru | +2.31 / +1.59 / -0.08 / +0.80 / +1.12 | +1.15 | 4/5 |
| SLEEPCTX | tf | -0.92 / +0.46 / +1.36 / -1.10 / +3.14 | +0.59 | 3/5 |

## (A) accumulation (gru, original order)

| level | comparison | d per pair | mean d | positive |
|---|---|---|---|---|
| SC | gru_orig − gru_orig_k10 | +14.24 / +14.47 / +13.47 | +14.06 | 3/3 |
| SC | gru_orig − gru_orig_k5 (report) | +32.62 / +32.97 / +32.30 | +32.63 | 3/3 |
| SLEEPCTX | gru_orig_L16 − gru_orig_L1 | +5.17 / +4.56 / -2.31 / +2.57 / +5.04 | +3.01 | 4/5 |
| SLEEPCTX | gru_orig_L16 − gru_orig_L4 (report) | +0.14 / -0.77 / -0.43 / -0.55 / +2.49 | +0.18 | 2/5 |

## PPT-style permuted test (report only): test − permuted test, models trained in the original order

| level | model | drop |
|---|---|---|
| SC | gru | +60.47 ± 1.30 |
| SC | tf | +2.21 ± 0.13 |
| MITBIH | gru | +65.78 ± 1.55 |
| MITBIH | tf | +52.57 ± 2.20 |
| SLEEPIN | gru | +2.46 ± 0.68 |
| SLEEPIN | tf | +0.17 ± 0.30 |
| SLEEPCTX | gru | +5.15 ± 1.48 |
| SLEEPCTX | tf | +1.08 ± 0.71 |

## Verdicts (pre-registered rules)

| level | (B) order | (A) accumulation | passes |
|---|---|---|---|
| SC | moderate (V_B +2.06 %p, gru, 3/3) | needed (V_A +14.06 %p, 3/3) | yes |
| MITBIH | moderate (V_B +2.65 %p, gru, 3/3) | — | yes (calibration) |
| SLEEPIN | weak (V_B -0.57 %p, tf, 2/5) | — | no |
| SLEEPCTX | moderate (V_B +1.15 %p, gru, 4/5) | needed (V_A +3.01 %p, 4/5) | yes |

Run time / peak GPU memory: SLEEPCTX 506 s / 2.5 GB, SLEEPIN 170 s / 1.7 GB, SC 74 s / 1.9 GB, MITBIH 63 s / 0.2 GB
