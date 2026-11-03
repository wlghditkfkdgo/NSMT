# H1 head (the head also reads the last neuron layer's charge) — mitbih

Settings (res, lr, mixer layers): muxlin ('add', 0.02, 2), Spikformer-style ('-', 0.005, 2); test macro-F1 of the last-epoch weights.

| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| G1 muxlin+H1 vs muxlin | +0.01 | -0.04, +0.01, +0.04 | no clear difference | 1.01 |  |
| G2 Spikformer-style+H1 vs Spikformer-style | -0.26 | -0.19, -0.57, -0.02 | no clear difference | 0.72 |  |
| G3 muxlin+H1 vs Spikformer-style+H1 | +0.47 | -0.07, +0.96, +0.51 | no clear difference | 1.38 | not met |

Report only — G1 gain − G2 gain per seed: +0.15, +0.58, +0.07 (mean +0.26 %p)

| condition | macro-F1 % | F1 N / S / V / F / Q % | energy µJ |
|---|---:|---|---:|
| Spikformer-style | 91.78 ± 0.25 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 9.61 |
| Spikformer-style + H1 | 91.52 ± 0.30 | 99.15 / 83.86 / 95.58 / 80.29 / 98.72 | 6.90 |
| muxlin | 91.99 ± 0.10 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 9.44 |
| muxlin + H1 | 91.99 ± 0.12 | 99.24 / 85.21 / 96.35 / 80.22 / 98.93 | 9.54 |
