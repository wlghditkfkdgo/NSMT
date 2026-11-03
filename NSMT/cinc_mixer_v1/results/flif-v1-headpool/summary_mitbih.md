# Head input forms (a zero-initialised head term on another input form of the last block) — mitbih

Settings (res, lr, mixer layers): muxlin ('add', 0.02, 2), Spikformer-style ('-', 0.005, 2); test macro-F1 of the last-epoch weights.

## max — W(any spike over the tokens)

| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| G1 muxlin+max vs muxlin | -0.23 | -0.31, -0.29, -0.09 | no clear difference | 0.95 |  |
| G2 Spikformer-style+max vs Spikformer-style | -0.32 | -0.66, -0.51, +0.23 | no clear difference | 0.80 |  |
| G3 muxlin+max vs Spikformer-style+max | +0.29 | +0.14, +0.61, +0.12 | no clear difference | 1.17 | not met |

Report only — G1 gain − G2 gain per seed: +0.36, +0.22, -0.32 (mean +0.09 %p)

| condition | macro-F1 % | F1 N / S / V / F / Q % | energy µJ |
|---|---:|---|---:|
| Spikformer-style | 91.78 ± 0.25 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 9.61 |
| Spikformer-style + max | 91.47 ± 0.21 | 99.17 / 83.42 / 95.98 / 79.75 / 99.01 | 7.66 |
| muxlin | 91.99 ± 0.10 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 9.44 |
| muxlin + max | 91.76 ± 0.13 | 99.21 / 85.31 / 96.13 / 79.25 / 98.88 | 8.94 |

## seg4 — W(4 contiguous token-segment means)

| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| G1 muxlin+seg4 vs muxlin | +0.06 | +0.11, -0.09, +0.15 | no clear difference | 0.95 |  |
| G2 Spikformer-style+seg4 vs Spikformer-style | +0.18 | +0.09, -0.00, +0.44 | no clear difference | 0.84 |  |
| G3 muxlin+seg4 vs Spikformer-style+seg4 | +0.08 | -0.20, +0.30, +0.15 | no clear difference | 1.11 | not met |

Report only — G1 gain − G2 gain per seed: +0.02, -0.09, -0.29 (mean -0.12 %p)

| condition | macro-F1 % | F1 N / S / V / F / Q % | energy µJ |
|---|---:|---|---:|
| Spikformer-style | 91.78 ± 0.25 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 9.61 |
| Spikformer-style + seg4 | 91.96 ± 0.19 | 99.22 / 84.95 / 96.13 / 80.61 / 98.89 | 8.04 |
| muxlin | 91.99 ± 0.10 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 9.44 |
| muxlin + seg4 | 92.04 ± 0.04 | 99.23 / 85.97 / 96.26 / 79.80 / 98.94 | 8.97 |

## layers — W(embedding and after-mixing token means)

| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| G1 muxlin+layers vs muxlin | -0.08 | +0.31, -0.10, -0.44 | no clear difference | 0.99 |  |
| G2 Spikformer-style+layers vs Spikformer-style | +0.00 | -0.20, +0.11, +0.10 | no clear difference | 0.90 |  |
| G3 muxlin+layers vs Spikformer-style+layers | +0.12 | +0.29, +0.18, -0.09 | no clear difference | 1.09 | not met |

Report only — G1 gain − G2 gain per seed: +0.50, -0.21, -0.53 (mean -0.08 %p)

| condition | macro-F1 % | F1 N / S / V / F / Q % | energy µJ |
|---|---:|---|---:|
| Spikformer-style | 91.78 ± 0.25 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 9.61 |
| Spikformer-style + layers | 91.78 ± 0.13 | 99.19 / 84.54 / 96.05 / 80.08 / 99.06 | 8.61 |
| muxlin | 91.99 ± 0.10 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 9.44 |
| muxlin + layers | 91.91 ± 0.29 | 99.25 / 85.83 / 96.18 / 79.25 / 99.03 | 9.36 |

## attn — learned token weights (reference only, real-valued)

| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| G1 muxlin+attn vs muxlin | -0.48 | -0.19, -0.64, -0.60 | no clear difference | 1.08 |  |
| G2 Spikformer-style+attn vs Spikformer-style | -0.18 | -0.46, -0.01, -0.07 | no clear difference | 0.81 |  |
| G3 muxlin+attn vs Spikformer-style+attn | -0.09 | +0.05, -0.24, -0.08 | no clear difference | 1.31 | not met |

Report only — G1 gain − G2 gain per seed: +0.27, -0.63, -0.52 (mean -0.29 %p)

| condition | macro-F1 % | F1 N / S / V / F / Q % | energy µJ |
|---|---:|---|---:|
| Spikformer-style | 91.78 ± 0.25 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 9.61 |
| Spikformer-style + attn | 91.60 ± 0.13 | 99.18 / 84.14 / 95.82 / 79.86 / 99.01 | 7.81 |
| muxlin | 91.99 ± 0.10 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 9.44 |
| muxlin + attn | 91.51 ± 0.14 | 99.19 / 85.10 / 96.15 / 78.27 / 98.83 | 10.23 |

## Adoption (pre-registered): CinC G1 or G2 better and the same G on MIT-BIH not worse; attn excluded

| form | CinC G1 | CinC G2 | MIT-BIH G1 | MIT-BIH G2 | adopted |
|---|---|---|---|---|---|
| max | worse | worse | no clear difference | no clear difference | no |
| seg4 | worse | worse | no clear difference | no clear difference | no |
| layers | worse | worse | no clear difference | no clear difference | no |
