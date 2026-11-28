# AD residual unified to IAND (one mixer layer per direction, lr 2e-3)

Rules: summarize_adiand.py docstring. 6 new runs (MSL / SMAP), commit `5a888c310`; PSM IAND = the 11-19 G5 runs.

| comparison | datasets | R (AUC-PR, %) | pairs higher | verdict | mean ΔAUC-PR | energy A/B syops (syn) | energy A/B syops + neuron | adoption bar |
|---|---|--:|--:|---|--:|--:|--:|---|
| J1 IAND-all (new MSL / SMAP) vs SF-de(tuned) | MSL / SMAP / PSM | +0.505 | 6/9 | **no clear difference** | +0.0006 | 1.0176 | 1.0117 | fail |
| J2 IAND-all (new MSL / SMAP) vs G5 (MSL / SMAP OR, PSM IAND) | MSL / SMAP | -1.386 | 1/6 | **no clear difference** | -0.0019 | 1.0318 | 1.0362 | fail |
| J0 (report) G5 (MSL / SMAP OR, PSM IAND) vs SF-de(tuned) | MSL / SMAP / PSM | +1.452 | 8/9 | **no clear difference** | +0.0019 | 0.9966 | 0.9880 | pass |

| comparison | per-dataset seed-mean r % (seeds 42 / 7 / 2026) ; ΔAUC-PR ; energy A/B syn |
|---|---|
| J1 | MSL +0.46 (+1.20, +0.91, -0.71), +0.0006, 1.014 ; SMAP +1.07 (+0.86, +2.76, -0.40), +0.0012, 1.029 ; PSM -0.02 (+0.10, +0.12, -0.28), -0.0001, 1.010 |
| J2 | MSL -1.75 (-1.71, -0.52, -3.01), -0.0026, 1.028 ; SMAP -1.03 (-1.72, +1.01, -2.37), -0.0012, 1.035 |
| J0 (report) | MSL +2.25 (+2.96, +1.43, +2.36), +0.0032, 0.986 ; SMAP +2.12 (+2.62, +1.72, +2.03), +0.0025, 0.994 ; PSM -0.02 (+0.10, +0.12, -0.28), -0.0001, 1.010 |

Recommendation (pre-registered rule): **keep G5**

## MSL (seed mean)

| arm | AUC-PR % | AUC-ROC % | PA-F1 % | raw F1 % | energy µJ syn | energy µJ syn + neuron | best epoch | time s |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| IAND-all (new MSL / SMAP) | 14.28 | 59.85 | 45.57 | 7.78 | 15.58 | 16.17 | 9.0 | 793 |
| G5 (MSL / SMAP OR, PSM IAND) | 14.53 | 60.86 | 41.29 | 7.60 | 15.15 | 15.66 | 9.0 | 536 |
| SF-de(tuned) | 14.21 | 59.36 | 45.92 | 7.68 | 15.36 | 15.99 | 9.0 | 466 |

## SMAP (seed mean)

| arm | AUC-PR % | AUC-ROC % | PA-F1 % | raw F1 % | energy µJ syn | energy µJ syn + neuron | best epoch | time s |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| IAND-all (new MSL / SMAP) | 11.70 | 44.69 | 67.76 | 2.72 | 7.39 | 7.95 | 9.0 | 2045 |
| G5 (MSL / SMAP OR, PSM IAND) | 11.82 | 44.75 | 68.35 | 2.92 | 7.13 | 7.64 | 9.0 | 1373 |
| SF-de(tuned) | 11.57 | 44.29 | 68.21 | 2.57 | 7.18 | 7.78 | 8.7 | 1157 |

## PSM (seed mean)

| arm | AUC-PR % | AUC-ROC % | PA-F1 % | raw F1 % | energy µJ syn | energy µJ syn + neuron | best epoch | time s |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| IAND-all (new MSL / SMAP) | 39.14 | 59.86 | 92.63 | 5.17 | 7.55 | 8.10 | 3.3 | 845 |
| G5 (MSL / SMAP OR, PSM IAND) | 39.14 | 59.86 | 92.63 | 5.17 | 7.55 | 8.10 | 3.3 | 845 |
| SF-de(tuned) | 39.15 | 59.82 | 92.31 | 5.38 | 7.48 | 8.09 | 8.0 | 861 |

Reproduction (report): PSM IAND seed 42 at commit `5a888c310` = the 11-19 row: val_mse_best 0.012448 vs 0.012448, aucpr 0.390383 vs 0.390383, syn_energy_uJ 7.552642 vs 7.552642, best_epoch 4 vs 4, epochs_run 8 vs 8 -> **same**
