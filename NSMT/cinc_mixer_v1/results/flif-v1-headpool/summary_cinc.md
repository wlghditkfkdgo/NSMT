# Head input forms (a zero-initialised head term on another input form of the last block) — cinc

Settings (res, lr, mixer layers): muxlin ('or', 0.001, 2), Spikformer-style ('-', 0.002, 2); CinC fold 0, test CinC score.

## max — W(any spike over the tokens)

| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| G1 muxlin+max vs muxlin | -1.07 | -1.93, -0.36, -0.92 | worse | 0.97 |  |
| G2 Spikformer-style+max vs Spikformer-style | -2.26 | -2.94, +1.63, -5.48 | worse | 0.88 |  |
| G3 muxlin+max vs Spikformer-style+max | +3.35 | +3.86, +0.55, +5.64 | better | 1.08 | met |

Report only — G1 gain − G2 gain per seed: +1.00, -1.99, +4.56 (mean +1.19 %p)

| condition | CinC score % | F1 class 0 / 1 / 2 / 3 % | energy µJ |
|---|---:|---|---:|
| Spikformer-style | 66.36 ± 1.04 | 82.95 / 61.18 / 54.94 / 50.43 | 131.51 |
| Spikformer-style + max | 64.10 ± 1.90 | 82.77 / 57.65 / 51.87 / 43.28 | 116.21 |
| muxlin | 68.52 ± 0.81 | 83.51 / 65.16 / 56.88 / 51.85 | 129.39 |
| muxlin + max | 67.45 ± 0.21 | 83.89 / 62.39 / 56.06 / 52.78 | 125.29 |

## seg4 — W(4 contiguous token-segment means)

| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| G1 muxlin+seg4 vs muxlin | -0.75 | -1.22, +0.27, -1.30 | worse | 0.97 |  |
| G2 Spikformer-style+seg4 vs Spikformer-style | -0.75 | -0.35, +1.48, -3.39 | worse | 0.87 |  |
| G3 muxlin+seg4 vs Spikformer-style+seg4 | +2.16 | +1.98, +1.33, +3.17 | better | 1.09 | met |

Report only — G1 gain − G2 gain per seed: -0.88, -1.22, +2.08 (mean -0.00 %p)

| condition | CinC score % | F1 class 0 / 1 / 2 / 3 % | energy µJ |
|---|---:|---|---:|
| Spikformer-style | 66.36 ± 1.04 | 82.95 / 61.18 / 54.94 / 50.43 | 131.51 |
| Spikformer-style + seg4 | 65.61 ± 1.08 | 82.97 / 58.31 / 55.54 / 47.28 | 114.17 |
| muxlin | 68.52 ± 0.81 | 83.51 / 65.16 / 56.88 / 51.85 | 129.39 |
| muxlin + seg4 | 67.77 ± 0.42 | 83.18 / 64.66 / 55.47 / 48.06 | 124.96 |

## layers — W(embedding and after-mixing token means)

| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| G1 muxlin+layers vs muxlin | -1.74 | -1.91, -2.47, -0.85 | worse | 1.00 |  |
| G2 Spikformer-style+layers vs Spikformer-style | -1.17 | -2.03, +0.67, -2.14 | worse | 0.92 |  |
| G3 muxlin+layers vs Spikformer-style+layers | +1.59 | +2.98, -0.60, +2.38 | better | 1.06 | met |

Report only — G1 gain − G2 gain per seed: +0.12, -3.15, +1.29 (mean -0.58 %p)

| condition | CinC score % | F1 class 0 / 1 / 2 / 3 % | energy µJ |
|---|---:|---|---:|
| Spikformer-style | 66.36 ± 1.04 | 82.95 / 61.18 / 54.94 / 50.43 | 131.51 |
| Spikformer-style + layers | 65.19 ± 0.43 | 82.57 / 58.91 / 54.09 / 50.93 | 121.29 |
| muxlin | 68.52 ± 0.81 | 83.51 / 65.16 / 56.88 / 51.85 | 129.39 |
| muxlin + layers | 66.78 ± 1.23 | 83.97 / 60.94 / 55.42 / 52.74 | 129.09 |

## attn — learned token weights (reference only, real-valued)

| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| G1 muxlin+attn vs muxlin | -5.36 | -4.79, -3.81, -7.48 | worse | 0.96 |  |
| G2 Spikformer-style+attn vs Spikformer-style | -5.63 | -5.44, -3.33, -8.13 | worse | 0.83 |  |
| G3 muxlin+attn vs Spikformer-style+attn | +2.44 | +3.51, +2.06, +1.74 | better | 1.13 | met |

Report only — G1 gain − G2 gain per seed: +0.65, -0.48, +0.65 (mean +0.27 %p)

| condition | CinC score % | F1 class 0 / 1 / 2 / 3 % | energy µJ |
|---|---:|---|---:|
| Spikformer-style | 66.36 ± 1.04 | 82.95 / 61.18 / 54.94 / 50.43 | 131.51 |
| Spikformer-style + attn | 60.72 ± 1.00 | 79.85 / 52.19 / 50.13 / 35.78 | 109.31 |
| muxlin | 68.52 ± 0.81 | 83.51 / 65.16 / 56.88 / 51.85 | 129.39 |
| muxlin + attn | 63.16 ± 1.53 | 82.23 / 53.59 / 53.66 / 35.57 | 123.59 |

