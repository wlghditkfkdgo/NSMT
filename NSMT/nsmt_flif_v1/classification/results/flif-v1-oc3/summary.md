# PTB order check — flif-v1-oc3 (sha 4320724aa)

Pre-registered: docs/MODI_PROJECT_LOG.md 2026-11-27. Test values in %, mean ± std over seeds 42 / 7 / 2026.

## PTB (judged)

| condition | macro-F1 | accuracy | permuted test macro-F1 / acc (report) |
|---|---|---|---|
| gru_orig | 84.04 ± 5.00 | 88.93 ± 2.90 | 53.48 / 74.29 |
| gru_shuf | 78.71 ± 3.62 | 85.42 ± 1.69 | — |
| tf_orig | 77.21 ± 2.94 | 84.47 ± 1.87 | 76.29 / 84.16 |
| tf_shuf | 76.11 ± 3.47 | 83.64 ± 1.60 | — |

| test | d per seed | mean | positive |
|---|---|---|---|
| (B) gru orig − shuf, macro-F1 | -3.25 / +13.62 / +5.63 | +5.33 | 2/3 |
| gru orig − shuf, accuracy (report) | -0.77 / +7.61 / +3.68 | +3.51 | 2/3 |
| gru test − permuted test, accuracy (PPT-style, report) | +11.26 / +16.59 / +16.06 | +14.63 | |
| (B) tf orig − shuf, macro-F1 | +4.37 / -1.01 / -0.05 | +1.10 | 1/3 |
| tf orig − shuf, accuracy (report) | +1.93 / -0.87 / +1.43 | +0.83 | 2/3 |
| tf test − permuted test, accuracy (PPT-style, report) | -0.60 / +0.06 / +1.48 | +0.31 | |
| best orig − best shuf, macro-F1 (report) | | +5.33 | |

**PTB: (B) moderate (V_B +5.33 %p, gru, 2/3) → weak candidate.**

## PTB30 (report only)

| condition | macro-F1 | accuracy | permuted test macro-F1 / acc (report) |
|---|---|---|---|
| gru_orig | 79.01 ± 3.51 | 85.89 ± 1.64 | 54.06 / 74.47 |
| gru_shuf | 78.71 ± 1.67 | 84.82 ± 0.74 | — |
| tf_orig | 74.18 ± 3.60 | 83.15 ± 1.80 | 71.42 / 81.98 |
| tf_shuf | 76.70 ± 3.40 | 84.01 ± 2.10 | — |

| test | d per seed | mean | positive |
|---|---|---|---|
| (B) gru orig − shuf, macro-F1 | +1.52 / -4.49 / +3.88 | +0.31 | 2/3 |
| gru orig − shuf, accuracy (report) | +1.73 / -1.14 / +2.62 | +1.07 | 2/3 |
| gru test − permuted test, accuracy (PPT-style, report) | +13.88 / +9.53 / +10.85 | +11.42 | |
| (B) tf orig − shuf, macro-F1 | -2.11 / -2.47 / -2.98 | -2.52 | 0/3 |
| tf orig − shuf, accuracy (report) | -1.21 / -0.46 / -0.92 | -0.87 | 0/3 |
| tf test − permuted test, accuracy (PPT-style, report) | +1.81 / +0.93 / +0.77 | +1.17 | |
| best orig − best shuf, macro-F1 (report) | | +0.31 | |

**PTB30: (B) weak (V_B +0.31 %p, gru, 2/3).**

Run time / peak GPU memory: PTB 63 s / 1.4 GB, PTB30 53 s / 1.2 GB
