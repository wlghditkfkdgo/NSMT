# Anomaly-detection tuning of the fusion MUX — stage final

Tuned settings: {"spkmx_lif_muxlin_or": {"MSL": {"lr": 0.002, "layers": 2, "res": "or"}, "SMAP": {"lr": 0.002, "layers": 1, "res": "add"}, "PSM": {"lr": 0.001, "layers": 3, "res": "add"}}, "spkmx_lif_sdlin_or": {"MSL": {"lr": 0.002, "layers": 2, "res": "or"}, "SMAP": {"lr": 0.002, "layers": 2, "res": "add"}, "PSM": {"lr": 0.001, "layers": 1, "res": "add"}}, "spk_ssa": {"MSL": {"lr": 0.00025, "layers": 2, "res": "-"}, "SMAP": {"lr": 0.002, "layers": 2, "res": "-"}, "PSM": {"lr": 0.0005, "layers": 2, "res": "-"}}}

| question | R (AUC-PR, %) | pairs higher | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| A1 muxlin(tuned) vs Spikformer-style(tuned) | +1.05 | 6/9 | no clear difference | 1.03 | not met |
| A2 muxlin(tuned) vs sdlin(tuned) | +0.61 | 7/9 | no clear difference | 1.03 |  |
| A3 sdlin(tuned) vs Spikformer-style(tuned) | +0.43 | 5/9 | no clear difference | 1.00 | not met |
| A4 muxlin(tuned) vs muxlin(default) | -1.25 | 1/9 | no clear difference | 1.05 |  |

| comparison | per-dataset mean r (seeds 42, 7, 2026) |
|---|---|
| A1 | MSL +1.43 (+2.95, +0.95, +0.38) . ; SMAP +1.92 (+2.02, +2.14, +1.59) . ; PSM -0.20 (-0.24, -0.23, -0.13) . |
| A2 | MSL +0.52 (+1.75, -0.09, -0.09) . ; SMAP +1.08 (+2.20, +0.28, +0.77) . ; PSM +0.23 (+0.35, +0.19, +0.16) . |
| A3 | MSL +0.90 (+1.18, +1.04, +0.48) . ; SMAP +0.83 (-0.17, +1.85, +0.82) . ; PSM -0.43 (-0.58, -0.42, -0.29) . |
| A4 | MSL -1.51 (-0.28, -1.23, -3.01) . ; SMAP -0.57 (+0.88, -0.78, -1.80) . ; PSM -1.66 (-1.67, -1.84, -1.49) . |

## MSL (seed mean ± std; report)

| condition | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | energy µJ | best epoch |
|---|---:|---:|---:|---:|---:|---:|
| Spikformer-style (tuned) | 14.46 ± 0.06 | 60.43 ± 0.09 | 7.66 ± 0.04 | 67.90 ± 4.90 | 14.50 | 9.0 |
| sdlin (tuned) | 14.59 ± 0.02 | 60.84 ± 0.14 | 7.71 ± 0.09 | 43.20 ± 2.90 | 14.41 | 9.0 |
| muxlin (tuned) | 14.67 ± 0.14 | 61.23 ± 0.30 | 7.81 ± 0.23 | 41.12 ± 0.16 | 14.64 | 8.7 |
| muxlin (default) | 14.89 ± 0.13 | 61.57 ± 0.25 | 7.89 ± 0.20 | 43.04 ± 3.03 | 14.48 | 9.0 |

## SMAP (seed mean ± std; report)

| condition | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | energy µJ | best epoch |
|---|---:|---:|---:|---:|---:|---:|
| Spikformer-style (tuned) | 11.59 ± 0.05 | 44.89 ± 0.10 | 2.47 ± 0.27 | 67.64 ± 0.76 | 7.18 | 8.7 |
| sdlin (tuned) | 11.69 ± 0.06 | 44.24 ± 0.12 | 3.12 ± 0.14 | 68.35 ± 0.05 | 7.32 | 8.7 |
| muxlin (tuned) | 11.81 ± 0.04 | 44.92 ± 0.08 | 3.03 ± 0.13 | 68.32 ± 0.02 | 7.06 | 8.7 |
| muxlin (default) | 11.88 ± 0.12 | 45.42 ± 0.94 | 2.76 ± 0.22 | 68.15 ± 0.10 | 7.22 | 9.0 |

## PSM (seed mean ± std; report)

| condition | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | energy µJ | best epoch |
|---|---:|---:|---:|---:|---:|---:|
| Spikformer-style (tuned) | 38.76 ± 0.04 | 59.54 ± 0.02 | 5.15 ± 0.05 | 93.75 ± 0.29 | 7.79 | 7.0 |
| sdlin (tuned) | 38.60 ± 0.08 | 59.41 ± 0.08 | 5.44 ± 0.07 | 95.18 ± 1.22 | 7.71 | 8.0 |
| muxlin (tuned) | 38.69 ± 0.06 | 59.45 ± 0.08 | 5.49 ± 0.04 | 93.78 ± 0.33 | 8.66 | 8.7 |
| muxlin (default) | 39.34 ± 0.01 | 60.12 ± 0.05 | 5.30 ± 0.02 | 92.60 ± 0.02 | 7.56 | 9.0 |
