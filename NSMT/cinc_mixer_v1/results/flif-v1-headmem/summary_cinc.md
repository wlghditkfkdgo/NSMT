# H1 head (the head also reads the last neuron layer's charge) — cinc

Settings (res, lr, mixer layers): muxlin ('or', 0.001, 2), Spikformer-style ('-', 0.002, 2); CinC fold 0, test CinC score.

| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| G1 muxlin+H1 vs muxlin | -1.33 | -3.16, +0.86, -1.70 | worse | 0.96 |  |
| G2 Spikformer-style+H1 vs Spikformer-style | -0.63 | -1.30, +1.15, -1.72 | worse | 0.82 |  |
| G3 muxlin+H1 vs Spikformer-style+H1 | +1.46 | +0.99, +2.26, +1.12 | better | 1.16 | met |

Report only — G1 gain − G2 gain per seed: -1.86, -0.29, +0.03 (mean -0.71 %p)

| condition | CinC score % | macro-F1 % | F1 class 0 / 1 / 2 / 3 % | energy µJ | best epoch:val |
|---|---:|---:|---|---:|---|
| Spikformer-style | 66.36 ± 1.04 | 62.38 | 82.95 / 61.18 / 54.94 / 50.43 | 131.51 | 23:0.716447, 20:0.717337, 26:0.704277 |
| Spikformer-style + H1 | 65.73 ± 0.32 | 61.67 | 82.91 / 60.03 / 54.25 / 49.50 | 107.84 | 23:0.699001, 25:0.706039, 23:0.720384 |
| muxlin | 68.52 ± 0.81 | 64.35 | 83.51 / 65.16 / 56.88 / 51.85 | 129.39 | 23:0.728381, 16:0.709590, 29:0.735785 |
| muxlin + H1 | 67.19 ± 0.86 | 62.70 | 83.83 / 62.59 / 55.14 / 49.26 | 124.60 | 18:0.705358, 21:0.708810, 26:0.701197 |
