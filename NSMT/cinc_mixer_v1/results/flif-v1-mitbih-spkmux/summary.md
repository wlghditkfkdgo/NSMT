# MIT-BIH extension of the fusion MUX `flif-v1-mitbih-spkmux` (LIF, residual OR; last-epoch weights, test set, seeds (42, 7, 2026))

| condition | n | macro-F1 % | accuracy % | F1 N | F1 S | F1 V | F1 F | F1 Q | total µJ | params | epochs | time s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spikformer-style (reused) | 3 | 92.22 ± 0.17 | 98.55 ± 0.02 | 99.24 | 85.28 | 96.04 | 81.38 | 99.17 | 9.55 | 904488 | 30 | 494 |
| sdlin (L) | 3 | 91.83 ± 0.26 | 98.58 ± 0.05 | 99.26 | 86.10 | 96.25 | 78.48 | 99.04 | 9.09 | 1200768 | 30 | 806 |
| muxlin (s_d ? L : F ^ R) | 3 | 91.90 ± 0.25 | 98.55 ± 0.01 | 99.24 | 85.74 | 96.38 | 79.24 | 98.89 | 8.89 | 1200768 | 30 | 851 |

## Pre-registered comparisons (macro-F1, %p, paired by seed)

| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| M1 muxlin vs Spikformer-style | -0.32 | -0.30, -0.22, -0.45 | no clear difference | 0.93 | not met |
| M2 muxlin vs sdlin | +0.07 | +0.05, +0.08, +0.09 | no clear difference | 0.98 |  |
| M3 sdlin vs Spikformer-style | -0.39 | -0.34, -0.30, -0.54 | no clear difference | 0.95 | not met |

RESULT rows (new): 6; SKIP lines: 0; baseline rows: 3.
