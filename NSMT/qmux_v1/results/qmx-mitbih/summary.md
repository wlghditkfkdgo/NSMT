# Query form with the bidirectional Neocortex — mitbih (qmx-mitbih)

Rules and sources: module docstring.

## MIT-BIH test macro-F1 (seed mean) and energy

| condition | macro-F1 % | accuracy % | F1 N / S / V / F / Q % | energy µJ | rho_neo |
|---|---:|---:|---|---:|---:|
| Spikformer-style (default, lr 0.01) | 92.22 ± 0.17 | 98.55 | 99.24 / 85.28 / 96.04 / 81.38 / 99.17 | 9.55 | nan |
| Spikformer-style (tuned, lr 0.005) | 91.78 ± 0.25 | 98.48 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 9.61 | nan |
| NSMT rerun | 91.58 ± 0.05 | 98.41 | 99.16 / 84.31 / 95.65 / 79.89 / 98.89 | 23.86 | 0.0931 |
| query lif (plain) | 91.84 ± 0.24 | 98.51 | 99.22 / 84.96 / 96.16 / 79.83 / 99.00 | 10.38 | 0.0733 |
| query lif, bidirectional muxlin, residual OR | 91.83 ± 0.09 | 98.54 | 99.23 / 85.49 / 96.20 / 79.10 / 99.12 | 11.58 | 0.0687 |
| query lif, bidirectional muxlin, residual IAND | 91.92 ± 0.07 | 98.54 | 99.23 / 85.45 / 96.28 / 79.63 / 99.04 | 11.21 | 0.0698 |
| query lif, bidirectional or, residual OR | 92.31 ± 0.38 | 98.59 | 99.27 / 85.53 / 96.30 / 81.36 / 99.07 | 11.68 | 0.0604 |
| query lif, bidirectional or, residual IAND | 92.23 ± 0.20 | 98.56 | 99.24 / 85.53 / 96.17 / 81.13 / 99.07 | 10.11 | 0.0663 |
| query flif (plain) | 92.21 ± 0.28 | 98.56 | 99.25 / 85.39 / 96.26 / 81.16 / 98.98 | 10.48 | 0.0673 |
| query flif, bidirectional muxlin, residual OR | 92.15 ± 0.14 | 98.57 | 99.25 / 85.75 / 96.15 / 80.49 / 99.09 | 11.60 | 0.0532 |
| query flif, bidirectional muxlin, residual IAND | 92.20 ± 0.25 | 98.53 | 99.23 / 85.24 / 95.98 / 81.42 / 99.11 | 11.40 | 0.0570 |
| query flif, bidirectional or, residual OR | 92.03 ± 0.15 | 98.53 | 99.23 / 85.33 / 96.11 / 80.42 / 99.05 | 11.43 | 0.0457 |
| query flif, bidirectional or, residual IAND | 92.11 ± 0.25 | 98.55 | 99.24 / 85.44 / 96.09 / 80.71 / 99.07 | 10.35 | 0.0499 |
| query pop_lif (plain) | 92.17 ± 0.13 | 98.57 | 99.26 / 86.18 / 96.08 / 80.36 / 98.98 | 11.10 | 0.0424 |
| query pop_lif, bidirectional muxlin, residual OR | 92.22 ± 0.20 | 98.58 | 99.27 / 86.17 / 96.13 / 80.51 / 99.02 | 13.65 | 0.0643 |
| query pop_lif, bidirectional muxlin, residual IAND | 92.16 ± 0.10 | 98.58 | 99.26 / 85.85 / 96.16 / 80.42 / 99.11 | 13.35 | 0.0485 |
| query pop_lif, bidirectional or, residual OR | 92.30 ± 0.16 | 98.53 | 99.23 / 85.74 / 96.14 / 81.66 / 98.72 | 12.63 | 0.0428 |
| query pop_lif, bidirectional or, residual IAND | 92.08 ± 0.15 | 98.54 | 99.23 / 85.34 / 96.13 / 80.67 / 99.03 | 11.46 | 0.0423 |
| query pop_flif (plain) | 92.02 ± 0.14 | 98.54 | 99.23 / 85.70 / 96.14 / 79.97 / 99.04 | 12.26 | 0.0623 |
| query pop_flif, bidirectional muxlin, residual OR | 92.11 ± 0.20 | 98.56 | 99.25 / 85.82 / 96.05 / 80.29 / 99.13 | 16.62 | 0.0569 |
| query pop_flif, bidirectional muxlin, residual IAND | 92.00 ± 0.13 | 98.56 | 99.25 / 85.57 / 96.09 / 80.00 / 99.10 | 16.27 | 0.0453 |
| query pop_flif, bidirectional or, residual OR | 92.26 ± 0.19 | 98.55 | 99.24 / 85.42 / 96.15 / 81.45 / 99.05 | 14.58 | 0.0510 |
| query pop_flif, bidirectional or, residual IAND | 91.98 ± 0.15 | 98.56 | 99.25 / 85.42 / 96.18 / 80.00 / 99.07 | 13.10 | 0.0424 |

## Pre-registered comparisons

| comparison | mean %p | seed 42, 7, 2026 | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| V1 lif OR: muxlin vs plain query | -0.01 | +0.29, -0.40, +0.09 | no clear difference | 1.12 | not met |
| V1 lif IAND: muxlin vs plain query | +0.09 | +0.37, -0.10, +0.00 | no clear difference | 1.08 | not met |
| V1 flif OR: muxlin vs plain query | -0.06 | -0.30, -0.27, +0.38 | no clear difference | 1.11 | not met |
| V1 flif IAND: muxlin vs plain query | -0.01 | -0.40, -0.23, +0.59 | no clear difference | 1.09 | not met |
| V1 pop_lif OR: muxlin vs plain query | +0.05 | +0.31, +0.08, -0.24 | no clear difference | 1.23 | not met |
| V1 pop_lif IAND: muxlin vs plain query | -0.01 | -0.03, +0.04, -0.04 | no clear difference | 1.20 | not met |
| V1 pop_flif OR: muxlin vs plain query | +0.09 | +0.17, -0.01, +0.11 | no clear difference | 1.36 | not met |
| V1 pop_flif IAND: muxlin vs plain query | -0.01 | +0.01, +0.30, -0.35 | no clear difference | 1.33 | not met |
| V2 lif OR: muxlin vs bidirectional OR | -0.48 | -0.95, -0.59, +0.11 | no clear difference | 0.99 | not met |
| V2 lif IAND: muxlin vs bidirectional OR | -0.30 | -0.60, -0.13, -0.17 | no clear difference | 1.11 | not met |
| V2 flif OR: muxlin vs bidirectional OR | +0.12 | -0.26, +0.43, +0.19 | no clear difference | 1.01 | not met |
| V2 flif IAND: muxlin vs bidirectional OR | +0.08 | -0.36, -0.06, +0.67 | no clear difference | 1.10 | not met |
| V2 pop_lif OR: muxlin vs bidirectional OR | -0.08 | +0.27, -0.44, -0.06 | no clear difference | 1.08 | not met |
| V2 pop_lif IAND: muxlin vs bidirectional OR | +0.08 | +0.08, -0.22, +0.38 | no clear difference | 1.17 | not met |
| V2 pop_flif OR: muxlin vs bidirectional OR | -0.16 | +0.18, -0.66, +0.01 | no clear difference | 1.14 | not met |
| V2 pop_flif IAND: muxlin vs bidirectional OR | +0.02 | -0.14, +0.23, -0.03 | no clear difference | 1.24 | not met |
| V3 lif OR: bidirectional OR vs plain query | +0.47 | +1.24, +0.20, -0.02 | no clear difference | 1.13 | not met |
| V3 lif IAND: bidirectional OR vs plain query | +0.39 | +0.97, +0.03, +0.17 | no clear difference | 0.97 | not met |
| V3 flif OR: bidirectional OR vs plain query | -0.18 | -0.04, -0.70, +0.20 | no clear difference | 1.09 | not met |
| V3 flif IAND: bidirectional OR vs plain query | -0.10 | -0.04, -0.16, -0.08 | no clear difference | 0.99 | not met |
| V3 pop_lif OR: bidirectional OR vs plain query | +0.13 | +0.04, +0.52, -0.18 | no clear difference | 1.14 | not met |
| V3 pop_lif IAND: bidirectional OR vs plain query | -0.09 | -0.11, +0.26, -0.43 | no clear difference | 1.03 | not met |
| V3 pop_flif OR: bidirectional OR vs plain query | +0.25 | -0.01, +0.65, +0.10 | no clear difference | 1.19 | not met |
| V3 pop_flif IAND: bidirectional OR vs plain query | -0.03 | +0.15, +0.07, -0.32 | no clear difference | 1.07 | not met |
| V4 lif OR: muxlin vs Spikformer-style (default) | -0.39 | -0.60, -0.52, -0.06 | no clear difference | 1.21 | not met |
| V4 lif IAND: muxlin vs Spikformer-style (default) | -0.30 | -0.52, -0.23, -0.14 | no clear difference | 1.17 | not met |
| V4 flif OR: muxlin vs Spikformer-style (default) | -0.08 | -0.48, +0.02, +0.23 | no clear difference | 1.21 | not met |
| V4 flif IAND: muxlin vs Spikformer-style (default) | -0.03 | -0.57, +0.06, +0.44 | no clear difference | 1.19 | not met |
| V4 pop_lif OR: muxlin vs Spikformer-style (default) | -0.00 | +0.08, -0.17, +0.08 | no clear difference | 1.43 | not met |
| V4 pop_lif IAND: muxlin vs Spikformer-style (default) | -0.06 | -0.26, -0.20, +0.28 | no clear difference | 1.40 | not met |
| V4 pop_flif OR: muxlin vs Spikformer-style (default) | -0.11 | -0.21, -0.41, +0.28 | no clear difference | 1.74 | not met |
| V4 pop_flif IAND: muxlin vs Spikformer-style (default) | -0.22 | -0.37, -0.11, -0.18 | no clear difference | 1.70 | not met |

## Report

| comparison | mean %p | seed 42, 7, 2026 | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| lif OR: muxlin vs Spikformer-style (tuned) | +0.05 | -0.29, -0.02, +0.44 | no clear difference | 1.21 | not met |
| lif OR: muxlin vs NSMT rerun | +0.25 | +0.18, +0.15, +0.41 | no clear difference | 0.49 | met |
| lif IAND: muxlin vs Spikformer-style (tuned) | +0.14 | -0.21, +0.28, +0.36 | no clear difference | 1.17 | not met |
| lif IAND: muxlin vs NSMT rerun | +0.34 | +0.25, +0.45, +0.33 | no clear difference | 0.47 | met |
| flif OR: muxlin vs Spikformer-style (tuned) | +0.36 | -0.16, +0.52, +0.73 | no clear difference | 1.21 | not met |
| flif OR: muxlin vs NSMT rerun | +0.57 | +0.30, +0.69, +0.70 | better | 0.49 | met |
| flif IAND: muxlin vs Spikformer-style (tuned) | +0.41 | -0.26, +0.56, +0.94 | no clear difference | 1.19 | not met |
| flif IAND: muxlin vs NSMT rerun | +0.61 | +0.20, +0.73, +0.91 | better | 0.48 | met |
| pop_lif OR: muxlin vs Spikformer-style (tuned) | +0.44 | +0.39, +0.34, +0.58 | no clear difference | 1.42 | not met |
| pop_lif OR: muxlin vs NSMT rerun | +0.64 | +0.86, +0.51, +0.55 | better | 0.57 | met |
| pop_lif IAND: muxlin vs Spikformer-style (tuned) | +0.38 | +0.06, +0.30, +0.78 | no clear difference | 1.39 | not met |
| pop_lif IAND: muxlin vs NSMT rerun | +0.58 | +0.52, +0.47, +0.75 | better | 0.56 | met |
| pop_flif OR: muxlin vs Spikformer-style (tuned) | +0.33 | +0.10, +0.09, +0.79 | no clear difference | 1.73 | not met |
| pop_flif OR: muxlin vs NSMT rerun | +0.53 | +0.56, +0.26, +0.75 | better | 0.70 | met |
| pop_flif IAND: muxlin vs Spikformer-style (tuned) | +0.22 | -0.05, +0.40, +0.32 | no clear difference | 1.69 | not met |
| pop_flif IAND: muxlin vs NSMT rerun | +0.42 | +0.41, +0.57, +0.29 | no clear difference | 0.68 | not met |
| lif: muxlin IAND vs muxlin OR | +0.10 | +0.08, +0.30, -0.09 | no clear difference | 0.97 | not met |
| lif: plain query vs Spikformer-style (default) | -0.39 | -0.89, -0.12, -0.15 | no clear difference | 1.09 | not met |
| flif: muxlin IAND vs muxlin OR | +0.05 | -0.10, +0.04, +0.20 | no clear difference | 0.98 | not met |
| flif: plain query vs Spikformer-style (default) | -0.01 | -0.18, +0.28, -0.15 | no clear difference | 1.10 | not met |
| pop_lif: muxlin IAND vs muxlin OR | -0.06 | -0.34, -0.04, +0.20 | no clear difference | 0.98 | not met |
| pop_lif: plain query vs Spikformer-style (default) | -0.05 | -0.23, -0.24, +0.32 | no clear difference | 1.16 | not met |
| pop_flif: muxlin IAND vs muxlin OR | -0.10 | -0.15, +0.31, -0.47 | no clear difference | 0.98 | not met |
| pop_flif: plain query vs Spikformer-style (default) | -0.20 | -0.38, -0.41, +0.17 | no clear difference | 1.28 | not met |

## Adoption (pre-registered): ETT V1 better and MIT-BIH V1 not worse

| neuron | residual | ETT V1 | MIT-BIH V1 | adopted |
|---|---|---|---|---|
| lif | OR | no clear difference (R -0.05 %) | no clear difference (-0.01 %p) | no |
| lif | IAND | no clear difference (R -0.27 %) | no clear difference (+0.09 %p) | no |
| flif | OR | no clear difference (R -0.34 %) | no clear difference (-0.06 %p) | no |
| flif | IAND | no clear difference (R -0.13 %) | no clear difference (-0.01 %p) | no |
| pop_lif | OR | no clear difference (R +0.21 %) | no clear difference (+0.05 %p) | no |
| pop_lif | IAND | no clear difference (R +0.22 %) | no clear difference (-0.01 %p) | no |
| pop_flif | OR | no clear difference (R +0.21 %) | no clear difference (+0.09 %p) | no |
| pop_flif | IAND | no clear difference (R +0.26 %) | no clear difference (-0.01 %p) | no |

Source files read (SHA256, first 16): ../../../Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/nsmt_flif_v1/classification/results/flif-v1-mitbih-query/raw.txt ae998bb0b463da68; ../../../Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/nsmt_flif_v1/forecasting/results/flif-v1-ett-query-m2/raw.txt 74158938b772ddfd; ../../../Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/nsmt_flif_v1/forecasting/results/flif-v1-ett-query/raw.txt 1d68020615f0db5e; ../../../Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/nsmt_flif_v1/forecasting/results/flif-v1-s2p10/raw.txt 85f9eb38238ea9fd; ../cinc_mixer_v1/results/flif-v1-ett-spkmux/baseline_spk_ssa_raw.txt 3e0c5936ec3a5656; ../cinc_mixer_v1/results/flif-v1-mitbih-spkmux/baseline_spk_ssa_raw.txt 05c707055bf82f54
