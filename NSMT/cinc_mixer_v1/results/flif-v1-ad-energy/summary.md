# AD energy test: (A) mean pooling, (B) one mixer layer per direction

Rules and sources: module docstring of summarize_ade.py. Seeds 42 / 7 / 2026; test AUC-PR of the best-validation weights.

Settings: {"(A) mean": {"MSL": "or / lr 0.002 / M 2 / mean", "SMAP": "add / lr 0.002 / M 1 / mean", "PSM": "add / lr 0.001 / M 3 / mean"}, "(B) M1": {"MSL": "add / lr 0.002 / M 1", "SMAP": "add / lr 0.002 / M 1", "PSM": "add / lr 0.002 / M 1"}, "(B) M1 0/1 residual": {"MSL": "or / lr 0.002 / M 1", "SMAP": "or / lr 0.002 / M 1", "PSM": "iand / lr 0.002 / M 1"}}

G5 run: yes

| comparison | R (AUC-PR, %) | pairs higher | verdict | energy A/B syops | paper bar (syops) | energy A/B current rule | paper bar (current rule) |
|---|---:|---|---|---:|---|---:|---|
| G1 (B) M1(tuned) vs SF-de(tuned) | +0.427 | 5/9 | no clear difference | 1.1857 | not met | 1.0111 | not met |
| G2 (B) M1(tuned) vs muxlin(tuned, 11-03) | -0.908 | 2/9 | no clear difference | 1.0013 |  | 0.9658 |  |
| G3 (A) mean vs SF-de(tuned) | +1.164 | 6/9 | no clear difference | 1.1789 | not met | 1.0500 | not met |
| G4 (A) mean vs muxlin(tuned, 11-03) | -0.197 | 3/9 | no clear difference | 0.9898 |  | 1.0037 |  |
| G5 (B) M1 0/1 residual vs SF-de(tuned) | +1.452 | 8/9 | no clear difference | 0.9880 | met | 0.9805 | met |
| (report) muxlin(tuned, 11-03) vs SF-de(tuned) | +1.368 | 6/9 | no clear difference | 1.1937 | not met | 1.0499 | not met |
| (report) muxlin(default) vs SF-de(tuned) | +2.648 | 9/9 | better | 1.0097 | not met | 1.0043 | not met |

| comparison | per-dataset mean r (seeds 42, 7, 2026) | energy A/B per dataset, syops (current) |
|---|---|---|
| G1 | MSL +0.24 (+1.52, +0.81, -1.60) . ; SMAP +2.06 (+2.22, +2.66, +1.31) b ; PSM -1.02 (-0.64, -1.30, -1.13) . | MSL 1.100 (0.994); SMAP 1.221 (0.982); PSM 1.236 (1.057) |
| G2 | MSL -2.89 (-3.46, -2.11, -3.09) w ; SMAP +0.00 (+0.00, +0.00, +0.00) . ; PSM +0.16 (+0.21, +0.37, -0.09) . | MSL 1.087 (0.984); SMAP 1.000 (1.000); PSM 0.917 (0.913) |
| G3 | MSL +3.09 (+3.53, +4.34, +1.42) b ; SMAP +1.73 (+1.12, +2.62, +1.46) . ; PSM -1.34 (-1.35, -1.22, -1.44) . | MSL 1.015 (1.013); SMAP 1.242 (1.050); PSM 1.280 (1.088) |
| G4 | MSL -0.12 (-1.56, +1.32, -0.12) . ; SMAP -0.32 (-1.07, -0.03, +0.14) . ; PSM -0.15 (-0.51, +0.45, -0.40) . | MSL 1.003 (1.003); SMAP 1.017 (1.068); PSM 0.949 (0.940) |
| G5 | MSL +2.25 (+2.96, +1.43, +2.36) b ; SMAP +2.12 (+2.62, +1.72, +2.03) b ; PSM -0.02 (+0.10, +0.12, -0.28) . | MSL 0.980 (0.974); SMAP 0.982 (0.964); PSM 1.002 (1.003) |
| (report) muxlin(tuned, 11-03) vs SF-de(tuned) | MSL +3.23 (+5.16, +2.98, +1.53) b ; SMAP +2.06 (+2.22, +2.66, +1.31) b ; PSM -1.18 (-0.85, -1.66, -1.04) . | MSL 1.012 (1.009); SMAP 1.221 (0.982); PSM 1.348 (1.158) |
| (report) muxlin(default) vs SF-de(tuned) | MSL +4.80 (+5.46, +4.27, +4.69) b ; SMAP +2.65 (+1.32, +3.47, +3.17) b ; PSM +0.49 (+0.83, +0.18, +0.45) . | MSL 1.000 (0.998); SMAP 1.015 (1.005); PSM 1.015 (1.010) |

## MSL (seed mean ± std; report)

| arm | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | energy µJ syops | energy µJ current rule | best epoch | time s (seeds) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| (A) mean | 14.65 ± 0.07 | 61.42 ± 0.27 | 7.43 ± 0.04 | 45.48 ± 2.33 | 16.23 | 14.69 | 9.0 | 684, 684, 624 |
| (B) M1 | 14.25 ± 0.08 | 59.38 ± 0.33 | 7.66 ± 0.08 | 45.79 ± 3.01 | 17.59 | 14.41 | 9.0 | 598, 583, 592 |
| (B) M1 0/1 residual | 14.53 ± 0.16 | 60.86 ± 0.53 | 7.60 ± 0.05 | 41.29 ± 0.24 | 15.66 | 14.12 | 9.0 | 590, 510, 507 |
| muxlin(tuned, 11-03) | 14.67 ± 0.14 | 61.23 ± 0.30 | 7.81 ± 0.23 | 41.12 ± 0.16 | 16.18 | 14.64 | 8.7 | 689, 713, 643 |
| muxlin(default) | 14.89 ± 0.13 | 61.57 ± 0.25 | 7.89 ± 0.20 | 43.04 ± 3.03 | 15.99 | 14.48 | 9.0 | 724, 679, 553 |
| SF-de(tuned) | 14.21 ± 0.12 | 59.36 ± 0.24 | 7.68 ± 0.07 | 45.92 ± 2.81 | 15.99 | 14.51 | 9.0 | 478, 479, 441 |

## SMAP (seed mean ± std; report)

| arm | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | energy µJ syops | energy µJ current rule | best epoch | time s (seeds) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| (A) mean | 11.77 ± 0.05 | 45.03 ± 0.09 | 2.66 ± 0.13 | 68.29 ± 0.06 | 9.66 | 7.54 | 8.7 | 1369, 1333, 1321 |
| (B) M1 | 11.81 ± 0.04 | 44.92 ± 0.08 | 3.03 ± 0.13 | 68.32 ± 0.02 | 9.50 | 7.06 | 8.7 | 1326, 1379, 1400 |
| (B) M1 0/1 residual | 11.82 ± 0.12 | 44.75 ± 0.56 | 2.92 ± 0.11 | 68.35 ± 0.02 | 7.64 | 6.93 | 9.0 | 1345, 1411, 1364 |
| muxlin(tuned, 11-03) | 11.81 ± 0.04 | 44.92 ± 0.08 | 3.03 ± 0.13 | 68.32 ± 0.02 | 9.50 | 7.06 | 8.7 | 3695, 3581, 2194 |
| muxlin(default) | 11.88 ± 0.12 | 45.42 ± 0.94 | 2.76 ± 0.22 | 68.15 ± 0.10 | 7.89 | 7.22 | 9.0 | 4127, 4141, 1886 |
| SF-de(tuned) | 11.57 ± 0.09 | 44.29 ± 0.29 | 2.57 ± 0.08 | 68.21 ± 0.10 | 7.78 | 7.18 | 8.7 | 1135, 1179, 1156 |

## PSM (seed mean ± std; report)

| arm | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | energy µJ syops | energy µJ current rule | best epoch | time s (seeds) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| (A) mean | 38.63 ± 0.13 | 59.36 ± 0.17 | 5.27 ± 0.05 | 94.09 ± 0.05 | 10.35 | 8.14 | 4.0 | 1400, 1282, 1312 |
| (B) M1 | 38.75 ± 0.01 | 59.57 ± 0.08 | 5.33 ± 0.09 | 93.38 ± 0.09 | 10.00 | 7.91 | 6.7 | 1076, 966, 1072 |
| (B) M1 0/1 residual | 39.14 ± 0.13 | 59.86 ± 0.07 | 5.17 ± 0.06 | 92.63 ± 0.03 | 8.10 | 7.51 | 3.3 | 919, 801, 814 |
| muxlin(tuned, 11-03) | 38.69 ± 0.06 | 59.45 ± 0.08 | 5.49 ± 0.04 | 93.78 ± 0.33 | 10.90 | 8.66 | 8.7 | 1552, 1660, 1593 |
| muxlin(default) | 39.34 ± 0.01 | 60.12 ± 0.05 | 5.30 ± 0.02 | 92.60 ± 0.02 | 8.21 | 7.56 | 9.0 | 1271, 1405, 1445 |
| SF-de(tuned) | 39.15 ± 0.11 | 59.82 ± 0.13 | 5.38 ± 0.01 | 92.31 ± 0.94 | 8.09 | 7.48 | 8.0 | 821, 878, 885 |

## Reproducibility (report): runs of this suite whose setting also ran on 11-03 (flif-v1-ad-spkmux-tune)

| setting | holdout | seed | value here | value 11-03 | same |
|---|---|---|---|---|---|
| add SMAP lr 0.002 M 1 | 0 | 2026 | aucpr 0.118207 | 0.118207 | yes |
| add SMAP lr 0.002 M 1 | 0 | 42 | aucpr 0.118517 | 0.118517 | yes |
| add SMAP lr 0.002 M 1 | 0 | 7 | aucpr 0.117605 | 0.117605 | yes |
| add SMAP lr 0.002 M 1 | 1 | 2026 | val_mse_best 0.012105 | 0.012105 | yes |
| add SMAP lr 0.002 M 1 | 1 | 42 | val_mse_best 0.011880 | 0.011880 | yes |
| add SMAP lr 0.002 M 1 | 1 | 7 | val_mse_best 0.011780 | 0.011780 | yes |
| iand SMAP lr 0.002 M 1 | 1 | 2026 | val_mse_best 0.018799 | 0.018799 | yes |
| iand SMAP lr 0.002 M 1 | 1 | 42 | val_mse_best 0.020116 | 0.020116 | yes |
| iand SMAP lr 0.002 M 1 | 1 | 7 | val_mse_best 0.018366 | 0.018366 | yes |
| or MSL lr 0.002 M 1 | 1 | 2026 | val_mse_best 0.048094 | 0.048094 | yes |
| or MSL lr 0.002 M 1 | 1 | 42 | val_mse_best 0.050482 | 0.050482 | yes |
| or MSL lr 0.002 M 1 | 1 | 7 | val_mse_best 0.050469 | 0.050469 | yes |
| or PSM lr 0.001 M 1 | 1 | 2026 | val_mse_best 0.015991 | 0.015991 | yes |
| or PSM lr 0.001 M 1 | 1 | 42 | val_mse_best 0.016048 | 0.016048 | yes |
| or PSM lr 0.001 M 1 | 1 | 7 | val_mse_best 0.015924 | 0.015924 | yes |
| or SMAP lr 0.002 M 1 | 1 | 2026 | val_mse_best 0.014848 | 0.014848 | yes |
| or SMAP lr 0.002 M 1 | 1 | 42 | val_mse_best 0.014686 | 0.014686 | yes |
| or SMAP lr 0.002 M 1 | 1 | 7 | val_mse_best 0.014076 | 0.014076 | yes |
