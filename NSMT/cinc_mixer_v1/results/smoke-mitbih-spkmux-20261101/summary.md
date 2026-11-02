# MIT-BIH extension of the fusion MUX `smoke-mitbih-spkmux-20261101` (LIF, residual OR; last-epoch weights, test set, seeds (42, 7, 2026))

| condition | n | macro-F1 % | accuracy % | F1 N | F1 S | F1 V | F1 F | F1 Q | total µJ | params | epochs | time s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Spikformer-style (reused) | 0 | | | | | | | | | | | |
| sdlin (L) | 1 | 74.82 | 94.03 | 96.84 | 54.28 | 82.21 | 46.67 | 94.12 | 9.06 | 1200768 | 1 | 39 |
| muxlin (s_d ? L : F ^ R) | 1 | 72.50 | 94.74 | 97.18 | 60.49 | 80.57 | 29.97 | 94.31 | 9.61 | 1200768 | 1 | 34 |

## Pre-registered comparisons (macro-F1, %p, paired by seed)

| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| M1 muxlin vs Spikformer-style | incomplete | | incomplete | | |
| M2 muxlin vs sdlin | incomplete | | incomplete | | |
| M3 sdlin vs Spikformer-style | incomplete | | incomplete | | |

RESULT rows (new): 2; SKIP lines: 0; baseline rows: 0.
