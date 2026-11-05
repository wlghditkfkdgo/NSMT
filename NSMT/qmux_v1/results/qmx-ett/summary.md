# Query form with the bidirectional Neocortex — ett (qmx-ett)

Rules and sources: module docstring.

## ETT test MSE (seed mean) and energy

| condition | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean MSE | energy µJ (4-cell mean) | rho_neo |
|---|---:|---:|---:|---:|---:|---:|---:|
| Spikformer-style (tuned) | 0.2903 | 0.4223 | 0.1767 | 0.4044 | 0.3234 | 10.48 | nan |
| NSMT rerun (same training rule) | 0.2896 | 0.4153 | 0.1744 | 0.4014 | 0.3202 | 65.42 | nan |
| query lif (plain) | 0.2925 | 0.4177 | 0.1773 | 0.4039 | 0.3229 | 11.40 | 0.0535 |
| query lif, bidirectional muxlin, residual OR | 0.2913 | 0.4178 | 0.1771 | 0.4050 | 0.3228 | 13.33 | 0.0538 |
| query lif, bidirectional muxlin, residual IAND | 0.2890 | 0.4176 | 0.1773 | 0.4044 | 0.3221 | 13.18 | 0.0579 |
| query lif, bidirectional or, residual OR | 0.2891 | 0.4136 | 0.1761 | 0.4047 | 0.3209 | 12.30 | 0.0421 |
| query lif, bidirectional or, residual IAND | 0.2885 | 0.4194 | 0.1771 | 0.4037 | 0.3222 | 12.00 | 0.0515 |
| query flif (plain) | 0.2930 | 0.4165 | 0.1770 | 0.4042 | 0.3227 | 11.76 | 0.0472 |
| query flif, bidirectional muxlin, residual OR | 0.2886 | 0.4173 | 0.1772 | 0.4037 | 0.3217 | 14.68 | 0.0476 |
| query flif, bidirectional muxlin, residual IAND | 0.2912 | 0.4183 | 0.1764 | 0.4043 | 0.3226 | 14.59 | 0.0506 |
| query flif, bidirectional or, residual OR | 0.2887 | 0.4190 | 0.1771 | 0.4046 | 0.3223 | 13.06 | 0.0359 |
| query flif, bidirectional or, residual IAND | 0.2924 | 0.4179 | 0.1771 | 0.4046 | 0.3230 | 12.78 | 0.0464 |
| query pop_lif (plain) | 0.2912 | 0.4135 | 0.1760 | 0.4028 | 0.3209 | 12.94 | 0.0190 |
| query pop_lif, bidirectional muxlin, residual OR | 0.2922 | 0.4157 | 0.1764 | 0.4018 | 0.3215 | 18.96 | 0.0395 |
| query pop_lif, bidirectional muxlin, residual IAND | 0.2913 | 0.4166 | 0.1762 | 0.4029 | 0.3217 | 18.91 | 0.0412 |
| query pop_lif, bidirectional or, residual OR | 0.2907 | 0.4168 | 0.1768 | 0.4018 | 0.3215 | 15.31 | 0.0174 |
| query pop_lif, bidirectional or, residual IAND | 0.2929 | 0.4188 | 0.1766 | 0.4015 | 0.3224 | 15.18 | 0.0209 |
| query pop_flif (plain) | 0.2890 | 0.4135 | 0.1765 | 0.4019 | 0.3203 | 15.74 | 0.0196 |
| query pop_flif, bidirectional muxlin, residual OR | 0.2923 | 0.4112 | 0.1769 | 0.4023 | 0.3207 | 29.07 | 0.0345 |
| query pop_flif, bidirectional muxlin, residual IAND | 0.2919 | 0.4139 | 0.1763 | 0.4024 | 0.3211 | 28.95 | 0.0322 |
| query pop_flif, bidirectional or, residual OR | 0.2877 | 0.4123 | 0.1768 | 0.4017 | 0.3196 | 20.94 | 0.0167 |
| query pop_flif, bidirectional or, residual IAND | 0.2900 | 0.4160 | 0.1761 | 0.4033 | 0.3214 | 20.77 | 0.0167 |

## Pre-registered comparisons

| comparison | R % | lower pairs | verdict | energy A/B | criterion (2) | per cell r % |
|---|---:|---:|---|---:|---|---|
| V1 lif OR: muxlin vs plain query | -0.05 | 7/12 | no clear difference | 1.17 | not met | -0.40, +0.02, -0.10, +0.29 |
| V1 lif IAND: muxlin vs plain query | -0.27 | 6/12 | no clear difference | 1.16 | not met | -1.21, -0.02, +0.01, +0.12 |
| V1 flif OR: muxlin vs plain query | -0.34 | 6/12 | no clear difference | 1.25 | not met | -1.49, +0.17, +0.07, -0.12 |
| V1 flif IAND: muxlin vs plain query | -0.13 | 5/12 | no clear difference | 1.24 | not met | -0.62, +0.42, -0.34, +0.03 |
| V1 pop_lif OR: muxlin vs plain query | +0.21 | 4/12 | no clear difference | 1.47 | not met | +0.36, +0.51, +0.22, -0.24 |
| V1 pop_lif IAND: muxlin vs plain query | +0.22 | 6/12 | no clear difference | 1.47 | not met | +0.03, +0.73, +0.10, +0.03 |
| V1 pop_flif OR: muxlin vs plain query | +0.21 | 5/12 | no clear difference | 1.86 | not met | +1.10, -0.57, +0.19, +0.10 |
| V1 pop_flif IAND: muxlin vs plain query | +0.26 | 4/12 | no clear difference | 1.85 | not met | +0.98, +0.09, -0.11, +0.11 |
| V2 lif OR: muxlin vs bidirectional OR | +0.62 | 3/12 | no clear difference | 1.08 | not met | +0.82, +1.01, +0.57, +0.08 |
| V2 lif IAND: muxlin vs bidirectional OR | +0.01 | 5/12 | no clear difference | 1.10 | not met | +0.17, -0.43, +0.14, +0.17 |
| V2 flif OR: muxlin vs bidirectional OR | -0.15 | 7/12 | no clear difference | 1.12 | not met | +0.00, -0.39, +0.02, -0.21 |
| V2 flif IAND: muxlin vs bidirectional OR | -0.19 | 9/12 | no clear difference | 1.14 | not met | -0.41, +0.09, -0.38, -0.08 |
| V2 pop_lif OR: muxlin vs bidirectional OR | +0.02 | 8/12 | no clear difference | 1.24 | not met | +0.54, -0.26, -0.23, +0.01 |
| V2 pop_lif IAND: muxlin vs bidirectional OR | -0.23 | 6/12 | no clear difference | 1.25 | not met | -0.55, -0.53, -0.19, +0.36 |
| V2 pop_flif OR: muxlin vs bidirectional OR | +0.38 | 5/12 | no clear difference | 1.39 | not met | +1.59, -0.27, +0.05, +0.16 |
| V2 pop_flif IAND: muxlin vs bidirectional OR | +0.00 | 7/12 | no clear difference | 1.39 | not met | +0.64, -0.50, +0.10, -0.23 |
| V3 lif OR: bidirectional OR vs plain query | -0.65 | 9/12 | no clear difference | 1.08 | not met | -1.17, -0.98, -0.66, +0.21 |
| V3 lif IAND: bidirectional OR vs plain query | -0.28 | 6/12 | no clear difference | 1.05 | not met | -1.38, +0.41, -0.12, -0.05 |
| V3 flif OR: bidirectional OR vs plain query | -0.19 | 6/12 | no clear difference | 1.11 | not met | -1.48, +0.58, +0.05, +0.09 |
| V3 flif IAND: bidirectional OR vs plain query | +0.07 | 5/12 | no clear difference | 1.09 | not met | -0.20, +0.33, +0.03, +0.11 |
| V3 pop_lif OR: bidirectional OR vs plain query | +0.20 | 5/12 | no clear difference | 1.19 | not met | -0.18, +0.80, +0.45, -0.25 |
| V3 pop_lif IAND: bidirectional OR vs plain query | +0.45 | 4/12 | no clear difference | 1.18 | not met | +0.57, +1.27, +0.29, -0.33 |
| V3 pop_flif OR: bidirectional OR vs plain query | -0.17 | 7/12 | no clear difference | 1.34 | not met | -0.46, -0.29, +0.14, -0.06 |
| V3 pop_flif IAND: bidirectional OR vs plain query | +0.26 | 3/12 | no clear difference | 1.33 | not met | +0.33, +0.59, -0.21, +0.33 |
| V4 lif OR: muxlin vs Spikformer-style (tuned) | -0.08 | 4/12 | no clear difference | 1.27 | not met | +0.35, -1.08, +0.24, +0.16 |
| V4 lif IAND: muxlin vs Spikformer-style (tuned) | -0.31 | 6/12 | no clear difference | 1.26 | not met | -0.47, -1.11, +0.35, -0.00 |
| V4 flif OR: muxlin vs Spikformer-style (tuned) | -0.42 | 8/12 | no clear difference | 1.40 | not met | -0.59, -1.20, +0.26, -0.17 |
| V4 flif IAND: muxlin vs Spikformer-style (tuned) | -0.21 | 6/12 | no clear difference | 1.39 | not met | +0.30, -0.95, -0.16, -0.01 |
| V4 pop_lif OR: muxlin vs Spikformer-style (tuned) | -0.43 | 8/12 | no clear difference | 1.81 | not met | +0.66, -1.57, -0.16, -0.63 |
| V4 pop_lif IAND: muxlin vs Spikformer-style (tuned) | -0.42 | 9/12 | no clear difference | 1.81 | not met | +0.31, -1.36, -0.28, -0.36 |
| V4 pop_flif OR: muxlin vs Spikformer-style (tuned) | -0.60 | 7/12 | no clear difference | 2.78 | not met | +0.66, -2.63, +0.09, -0.51 |
| V4 pop_flif IAND: muxlin vs Spikformer-style (tuned) | -0.54 | 9/12 | no clear difference | 2.77 | not met | +0.53, -1.99, -0.22, -0.50 |

## Report

| comparison | R % | lower pairs | verdict | energy A/B | criterion (2) | per cell r % |
|---|---:|---:|---|---:|---|---|
| lif OR: muxlin vs Spikformer-style (default) | -0.13 | 6/12 | no clear difference | 1.28 | not met | +0.35, -1.08, -0.19, +0.39 |
| lif OR: muxlin vs NSMT rerun | +0.92 | 1/12 | no clear difference | 0.20 | met | +0.62, +0.59, +1.57, +0.90 |
| lif IAND: muxlin vs Spikformer-style (default) | -0.36 | 7/12 | no clear difference | 1.26 | not met | -0.47, -1.11, -0.08, +0.23 |
| lif IAND: muxlin vs NSMT rerun | +0.69 | 3/12 | no clear difference | 0.20 | met | -0.20, +0.55, +1.69, +0.73 |
| flif OR: muxlin vs Spikformer-style (default) | -0.47 | 9/12 | no clear difference | 1.41 | not met | -0.59, -1.20, -0.16, +0.06 |
| flif OR: muxlin vs NSMT rerun | +0.58 | 2/12 | no clear difference | 0.23 | met | -0.32, +0.47, +1.60, +0.57 |
| flif IAND: muxlin vs Spikformer-style (default) | -0.26 | 8/12 | no clear difference | 1.40 | not met | +0.30, -0.95, -0.58, +0.22 |
| flif IAND: muxlin vs NSMT rerun | +0.79 | 3/12 | no clear difference | 0.22 | met | +0.57, +0.72, +1.17, +0.72 |
| pop_lif OR: muxlin vs Spikformer-style (default) | -0.48 | 8/12 | no clear difference | 1.82 | not met | +0.66, -1.57, -0.58, -0.40 |
| pop_lif OR: muxlin vs NSMT rerun | +0.57 | 3/12 | no clear difference | 0.29 | met | +0.93, +0.09, +1.17, +0.10 |
| pop_lif IAND: muxlin vs Spikformer-style (default) | -0.47 | 10/12 | no clear difference | 1.81 | not met | +0.31, -1.36, -0.70, -0.13 |
| pop_lif IAND: muxlin vs NSMT rerun | +0.58 | 4/12 | no clear difference | 0.29 | met | +0.59, +0.31, +1.05, +0.37 |
| pop_flif OR: muxlin vs Spikformer-style (default) | -0.65 | 9/12 | no clear difference | 2.79 | not met | +0.66, -2.63, -0.34, -0.27 |
| pop_flif OR: muxlin vs NSMT rerun | +0.40 | 5/12 | no clear difference | 0.45 | met | +0.94, -0.99, +1.42, +0.23 |
| pop_flif IAND: muxlin vs Spikformer-style (default) | -0.59 | 9/12 | no clear difference | 2.78 | not met | +0.53, -1.99, -0.64, -0.27 |
| pop_flif IAND: muxlin vs NSMT rerun | +0.45 | 5/12 | no clear difference | 0.44 | met | +0.80, -0.33, +1.11, +0.23 |
| lif: muxlin IAND vs muxlin OR | -0.23 | 6/12 | no clear difference | 0.99 | not met | -0.82, -0.04, +0.11, -0.16 |
| lif: plain query vs Spikformer-style (tuned) | -0.03 | 4/12 | no clear difference | 1.09 | not met | +0.75, -1.09, +0.34, -0.13 |
| flif: muxlin IAND vs muxlin OR | +0.22 | 5/12 | no clear difference | 0.99 | not met | +0.90, +0.25, -0.42, +0.15 |
| flif: plain query vs Spikformer-style (tuned) | -0.07 | 6/12 | no clear difference | 1.12 | not met | +0.93, -1.37, +0.19, -0.05 |
| pop_lif: muxlin IAND vs muxlin OR | +0.01 | 7/12 | no clear difference | 1.00 | not met | -0.33, +0.23, -0.12, +0.27 |
| pop_lif: plain query vs Spikformer-style (tuned) | -0.63 | 8/12 | no clear difference | 1.23 | not met | +0.30, -2.07, -0.38, -0.39 |
| pop_flif: muxlin IAND vs muxlin OR | +0.06 | 6/12 | no clear difference | 1.00 | not met | -0.12, +0.66, -0.30, +0.01 |
| pop_flif: plain query vs Spikformer-style (tuned) | -0.81 | 8/12 | no clear difference | 1.50 | not met | -0.44, -2.08, -0.10, -0.60 |

## Residual choice (validation only: mean over cells of best validation L1, IAND / OR)

| neuron | muxlin score | chosen |
|---|---:|---|
| lif | 1.0000 | IAND |
| flif | 0.9990 | IAND |
| pop_lif | 0.9983 | IAND |
| pop_flif | 0.9991 | IAND |

Source files read (SHA256, first 16): ../../../Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/nsmt_flif_v1/forecasting/results/flif-v1-ett-query-m2/raw.txt 74158938b772ddfd; ../../../Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/nsmt_flif_v1/forecasting/results/flif-v1-ett-query/raw.txt 1d68020615f0db5e; ../../../Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/nsmt_flif_v1/forecasting/results/flif-v1-s2p10/raw.txt 85f9eb38238ea9fd; ../cinc_mixer_v1/results/flif-v1-ett-spkmux/baseline_spk_ssa_raw.txt 3e0c5936ec3a5656
