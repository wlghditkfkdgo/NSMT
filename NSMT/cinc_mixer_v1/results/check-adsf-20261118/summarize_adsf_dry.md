# AD: SpkMixer (muxlin) vs SF-de

Rules: module docstring of summarize_adsf.py. Seeds 42 / 7 / 2026; test AUC-PR of the best-validation weights.

SF-de selected lr: {"MSL": "0.00025", "SMAP": "0.002", "PSM": "0.0005"}

| comparison | R (AUC-PR, %) | pairs higher | verdict | energy A/B | paper bar (R > 0 and energy < 1) |
|---|---:|---|---|---:|---|
| E1 muxlin(tuned) vs SF-de(tuned) | +1.05 | 6/9 | no clear difference | 1.03 | not met (R +1.049, energy 1.0347) |
| (report) muxlin(default) vs SF-de(tuned) | +2.33 | 9/9 | better | 0.99 | met (R +2.328, energy 0.9911) |
| (report) SF-de(tuned) vs SF-sj(tuned) | +0.00 | 0/9 | no clear difference | 1.00 |  |
| (report) muxlin(tuned) vs SF-sj(tuned) (= 11-03 A1) | +1.05 | 6/9 | no clear difference | 1.03 |  |

| comparison | per-dataset mean r (seeds 42, 7, 2026) |
|---|---|
| E1 muxlin(tuned) vs SF-de(tuned) | MSL +1.43 (+2.95, +0.95, +0.38) . ; SMAP +1.92 (+2.02, +2.14, +1.59) . ; PSM -0.20 (-0.24, -0.23, -0.13) . |
| (report) muxlin(default) vs SF-de(tuned) | MSL +2.98 (+3.24, +2.21, +3.50) b ; SMAP +2.51 (+1.13, +2.95, +3.45) b ; PSM +1.49 (+1.45, +1.64, +1.38) . |
| (report) SF-de(tuned) vs SF-sj(tuned) | MSL +0.00 (+0.00, +0.00, +0.00) . ; SMAP +0.00 (+0.00, +0.00, +0.00) . ; PSM +0.00 (+0.00, +0.00, +0.00) . |
| (report) muxlin(tuned) vs SF-sj(tuned) | MSL +1.43 (+2.95, +0.95, +0.38) . ; SMAP +1.92 (+2.02, +2.14, +1.59) . ; PSM -0.20 (-0.24, -0.23, -0.13) . |

## MSL (seed mean ± std; report)

| model | lr | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | energy µJ | best epoch | time s (seeds) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SF-de (tuned) | 0.00025 | 14.46 ± 0.06 | 60.43 ± 0.09 | 7.66 ± 0.04 | 67.90 ± 4.90 | 14.50 | 9.0 | 217, 213, 234 |
| SF-sj (tuned) | 0.00025 | 14.46 ± 0.06 | 60.43 ± 0.09 | 7.66 ± 0.04 | 67.90 ± 4.90 | 14.50 | 9.0 | 217, 213, 234 |
| muxlin (tuned) | 0.002 | 14.67 ± 0.14 | 61.23 ± 0.30 | 7.81 ± 0.23 | 41.12 ± 0.16 | 14.64 | 8.7 | 689, 713, 643 |
| muxlin (default) | 0.0005 | 14.89 ± 0.13 | 61.57 ± 0.25 | 7.89 ± 0.20 | 43.04 ± 3.03 | 14.48 | 9.0 | 724, 679, 553 |

## SMAP (seed mean ± std; report)

| model | lr | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | energy µJ | best epoch | time s (seeds) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SF-de (tuned) | 0.002 | 11.59 ± 0.05 | 44.89 ± 0.10 | 2.47 ± 0.27 | 67.64 ± 0.76 | 7.18 | 8.7 | 2304, 2304, 814 |
| SF-sj (tuned) | 0.002 | 11.59 ± 0.05 | 44.89 ± 0.10 | 2.47 ± 0.27 | 67.64 ± 0.76 | 7.18 | 8.7 | 2304, 2304, 814 |
| muxlin (tuned) | 0.002 | 11.81 ± 0.04 | 44.92 ± 0.08 | 3.03 ± 0.13 | 68.32 ± 0.02 | 7.06 | 8.7 | 3695, 3581, 2194 |
| muxlin (default) | 0.0005 | 11.88 ± 0.12 | 45.42 ± 0.94 | 2.76 ± 0.22 | 68.15 ± 0.10 | 7.22 | 9.0 | 4127, 4141, 1886 |

## PSM (seed mean ± std; report)

| model | lr | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | energy µJ | best epoch | time s (seeds) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| SF-de (tuned) | 0.0005 | 38.76 ± 0.04 | 59.54 ± 0.02 | 5.15 ± 0.05 | 93.75 ± 0.29 | 7.79 | 7.0 | 324, 326, 351 |
| SF-sj (tuned) | 0.0005 | 38.76 ± 0.04 | 59.54 ± 0.02 | 5.15 ± 0.05 | 93.75 ± 0.29 | 7.79 | 7.0 | 324, 326, 351 |
| muxlin (tuned) | 0.001 | 38.69 ± 0.06 | 59.45 ± 0.08 | 5.49 ± 0.04 | 93.78 ± 0.33 | 8.66 | 8.7 | 1552, 1660, 1593 |
| muxlin (default) | 0.0005 | 39.34 ± 0.01 | 60.12 ± 0.05 | 5.30 ± 0.02 | 92.60 ± 0.02 | 7.56 | 9.0 | 1271, 1405, 1445 |
