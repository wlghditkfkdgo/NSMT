# Second diagnostic batch — flif-v1-oc2 (sha 1d6dafe74)

Pre-registered: docs/MODI_PROJECT_LOG.md 2026-11-26 (evening). Values in %, mean ± std over seeds (MBIP) or folds (SLEEPPSD); differences in %p, paired.

## SLEEPPSD (Sleep-EDF, epoch-spectrum tokens)

| condition | macro-F1 | accuracy | permuted test MF1 (report) |
|---|---|---|---|
| gru_orig_L16 | 75.92 ± 3.91 | 81.72 ± 3.93 | 71.23 |
| gru_shuf_L16 | 74.33 ± 3.32 | 81.04 ± 3.05 | — |
| tf_orig_L16 | 73.92 ± 3.04 | 80.74 ± 3.43 | 72.62 |
| tf_shuf_L16 | 74.41 ± 2.95 | 80.80 ± 3.28 | — |
| gru_orig_L4 | 73.55 ± 3.39 | 80.74 ± 3.34 | 72.85 |
| gru_orig_L1 | 69.24 ± 4.29 | 78.30 ± 3.76 | 69.24 |

| test | d per fold | mean | positive | verdict |
|---|---|---|---|---|
| (B) gru orig − shuf | +2.63 / +0.73 / +1.50 / +2.84 / +0.29 | +1.60 | 5/5 | |
| (B) tf orig − shuf | +2.13 / -5.60 / +0.04 / +0.70 / +0.28 | -0.49 | 4/5 | **moderate** (V_B +1.60, gru) |
| (A) L16 − L1 | +8.46 / +4.78 / +5.59 / +6.63 / +7.93 | +6.68 | 5/5 | **needed** |
| L16 − L4 (report) | +1.61 / +4.13 / +3.10 / +2.73 / +0.29 | +2.37 | 5/5 | |
| best orig − best shuf (report) | | +1.51 | | |

**SLEEPPSD passes: yes** → (나) SNN comparison may start.

Front end (report only, macro-F1): gru_orig_L16 CNN 78.22 vs spectrum 75.92; tf_orig_L16 CNN 77.37 vs spectrum 73.92; gru_orig_L1 CNN 75.21 vs spectrum 69.24

## MBIP (MIT-BIH DS1 → DS2, token = one beat)

| condition | S F1 | S Se | S +P | V F1 | macro-F1 (4) | S Se 232 | S Se 222 | permuted test macro-F1 (report) |
|---|---|---|---|---|---|---|---|---|
| gru_orig_L17 | 4.07 ± 0.99 | 4.82 | 4.62 | 69.21 | 42.62 ± 2.69 | 0.17 | 15.79 | 42.26 |
| gru_shuf_L17 | 3.07 ± 1.00 | 3.43 | 2.88 | 61.87 | 39.04 ± 2.03 | 0.07 | 14.04 | — |
| tf_orig_L17 | 6.27 ± 1.46 | 8.51 | 5.06 | 64.67 | 40.44 ± 1.18 | 0.36 | 43.06 | 40.45 |
| tf_shuf_L17 | 4.51 ± 1.45 | 9.18 | 3.13 | 72.92 | 42.49 ± 0.92 | 0.12 | 56.30 | — |
| gru_orig_L5 | 5.06 ± 1.44 | 9.44 | 3.88 | 68.26 | 40.30 ± 0.47 | 0.07 | 43.38 | 40.22 |
| gru_orig_L2 | 3.56 ± 0.53 | 5.55 | 3.47 | 65.02 | 42.98 ± 0.88 | 0.22 | 23.60 | 42.98 |
| gru_orig_L1 | 5.06 ± 1.27 | 9.55 | 3.76 | 45.14 | 32.87 ± 4.42 | 0.22 | 56.94 | 32.87 |
| gru_orig_L1_rr | 4.79 ± 1.39 | 7.28 | 3.63 | 51.73 | 35.09 ± 1.14 | 0.27 | 35.41 | 35.09 |

| test (S F1) | d per seed | mean | positive | verdict |
|---|---|---|---|---|
| (A) gru L17 − L1 | -2.29 / -0.42 / -0.29 | -1.00 | 0/3 | **not shown** |
| (B) gru orig − shuf (L17) | -0.18 / +1.16 / +2.01 | +1.00 | 2/3 | |
| (B) tf orig − shuf (L17) | +0.71 / +0.68 / +3.89 | +1.76 | 3/3 | **moderate** (V_B +1.76, tf) |
| (C) gru L17 − L1 + two numbers | +0.08 / -0.84 / -1.43 | -0.73 | 1/3 | **not beaten** |
| L17 − L5 (report) | -0.68 / -0.64 / -1.67 | -1.00 | 0/3 | |
| L5 − L2 (report) | +1.34 / +0.47 / +2.69 | +1.50 | 3/3 | |
| L2 − L1 (report) | -2.94 / -0.25 / -1.32 | -1.50 | 0/3 | |
| L1 + two numbers − L1 (report) | -2.37 / +0.42 / +1.14 | -0.27 | 2/3 | |
| best orig − best shuf (report) | | +1.76 | | |

| same tests on macro-F1 (4 classes), report only | (A) | (B) gru | (B) tf | (C) |
|---|---|---|---|---|
| mean d | +9.74 (3/3) | +3.58 (2/3) | -2.05 (0/3) | +7.53 (3/3) |

**MBIP reading (pre-registered): past beats not shown to help.**

Run time / peak GPU memory: MBIP 34 s / 0.2 GB, SLEEPPSD 25 s / 0.2 GB
