# H1 head (the head also reads the last neuron layer's charge) — ad

Settings: {"MSL": {"muxlin": ["or", 0.002, 2], "spk_ssa": ["-", 0.00025, 2]}, "SMAP": {"muxlin": ["add", 0.002, 1], "spk_ssa": ["-", 0.002, 2]}, "PSM": {"muxlin": ["add", 0.001, 3], "spk_ssa": ["-", 0.0005, 2]}}; whole training file, test AUC-PR; r = 100 (AUC-PR_A − AUC-PR_B) / AUC-PR_B.

| question | R (AUC-PR, %) | pairs higher | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| G1 muxlin+H1 vs muxlin | +0.87 | 5/9 | no clear difference | 1.08 |  |
| G2 Spikformer-style+H1 vs Spikformer-style | +0.22 | 5/9 | no clear difference | 1.09 |  |
| G3 muxlin+H1 vs Spikformer-style+H1 | +1.70 | 7/9 | no clear difference | 1.03 | not met |

| comparison | per-dataset mean r (seeds 42, 7, 2026) |
|---|---|
| G1 | MSL +0.37 (-1.48, +0.98, +1.60) . ; SMAP +2.78 (+0.44, +1.66, +6.25) b ; PSM -0.54 (-0.51, -0.46, -0.64) . |
| G2 | MSL +0.51 (+0.49, +0.77, +0.29) . ; SMAP +0.64 (-0.31, +0.02, +2.20) . ; PSM -0.49 (-0.30, -0.34, -0.83) . |
| G3 | MSL +1.27 (+0.93, +1.16, +1.70) . ; SMAP +4.08 (+2.79, +3.82, +5.61) b ; PSM -0.25 (-0.45, -0.35, +0.06) . |

Report only — G1 relative change − G2 relative change per pair (aucpr): -1.97, +0.21, +1.31, +0.75, +1.65, +4.04, -0.21, -0.12, +0.19 (mean +0.65 %)

## MSL (seed mean ± std, %; report)

| condition | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | energy µJ | best epoch |
|---|---:|---:|---:|---:|---:|---:|
| Spikformer-style | 14.46 ± 0.06 | 60.43 ± 0.09 | 7.66 ± 0.04 | 67.90 ± 4.90 | 14.50 | 9.0 |
| Spikformer-style + H1 | 14.54 ± 0.04 | 61.19 ± 0.14 | 7.45 ± 0.11 | 55.69 ± 11.25 | 15.97 | 9.0 |
| muxlin | 14.67 ± 0.14 | 61.23 ± 0.30 | 7.81 ± 0.23 | 41.12 ± 0.16 | 14.64 | 8.7 |
| muxlin + H1 | 14.72 ± 0.08 | 61.41 ± 0.20 | 7.83 ± 0.13 | 68.30 ± 5.08 | 16.03 | 8.7 |

## SMAP (seed mean ± std, %; report)

| condition | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | energy µJ | best epoch |
|---|---:|---:|---:|---:|---:|---:|
| Spikformer-style | 11.59 ± 0.05 | 44.89 ± 0.10 | 2.47 ± 0.27 | 67.64 ± 0.76 | 7.18 | 8.7 |
| Spikformer-style + H1 | 11.66 ± 0.16 | 44.25 ± 0.18 | 3.15 ± 0.58 | 68.61 ± 0.12 | 7.80 | 9.0 |
| muxlin | 11.81 ± 0.04 | 44.92 ± 0.08 | 3.03 ± 0.13 | 68.32 ± 0.02 | 7.06 | 8.7 |
| muxlin + H1 | 12.14 ± 0.30 | 45.14 ± 0.27 | 3.61 ± 0.45 | 68.86 ± 0.20 | 7.72 | 8.7 |

## PSM (seed mean ± std, %; report)

| condition | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | energy µJ | best epoch |
|---|---:|---:|---:|---:|---:|---:|
| Spikformer-style | 38.76 ± 0.04 | 59.54 ± 0.02 | 5.15 ± 0.05 | 93.75 ± 0.29 | 7.79 | 7.0 |
| Spikformer-style + H1 | 38.57 ± 0.06 | 59.36 ± 0.02 | 5.42 ± 0.01 | 93.85 ± 0.30 | 8.33 | 7.7 |
| muxlin | 38.69 ± 0.06 | 59.45 ± 0.08 | 5.49 ± 0.04 | 93.78 ± 0.33 | 8.66 | 8.7 |
| muxlin + H1 | 38.48 ± 0.03 | 59.20 ± 0.08 | 5.50 ± 0.04 | 94.71 ± 0.98 | 9.23 | 9.0 |
