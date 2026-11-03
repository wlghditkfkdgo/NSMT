# H1 head (the head also reads the last neuron layer's charge) — ett

Settings: muxlin res {'ETTh2': 'iand', 'ETTm2': 'iand'} lr {'ETTh2': 0.0005, 'ETTm2': 0.0001} layers {'ETTh2': 1, 'ETTm2': 1}; Spikformer-style lr {'ETTh2': 0.0005, 'ETTm2': 0.00025}; cells (('ETTh2', 96), ('ETTm2', 96)); test MSE of the best-validation weights; r = 100 (MSE_A − MSE_B) / MSE_B.

| question | R (MSE, %) | pairs lower | verdict | energy A/B | criterion (2) | R on MAE |
|---|---:|---|---|---:|---|---:|
| G1 muxlin+H1 vs muxlin | +2.81 | 0/6 | worse | 1.48 |  | +2.21 |
| G2 Spikformer-style+H1 vs Spikformer-style | +0.63 | 2/6 | no clear difference | 1.47 |  | +0.84 |
| G3 muxlin+H1 vs Spikformer-style+H1 | +3.25 | 0/6 | worse | 0.98 | not met | +2.12 |

| comparison | per-cell mean r (seeds 42, 7, 2026) |
|---|---|
| G1 | ETTh2 H96 +4.38 (+0.70, +6.89, +5.56) w ; ETTm2 H96 +1.23 (+1.99, +1.03, +0.68) w |
| G2 | ETTh2 H96 +0.90 (-0.89, +1.40, +2.18) . ; ETTm2 H96 +0.35 (+1.16, -0.30, +0.19) . |
| G3 | ETTh2 H96 +5.16 (+4.02, +5.83, +5.63) w ; ETTm2 H96 +1.35 (+1.01, +1.77, +1.26) w |

Report only — G1 relative change − G2 relative change per pair (mse): +1.59, +5.49, +3.38, +0.82, +1.33, +0.48 (mean +2.18 %)

| condition | ETTh2 H96 MSE / MAE / energy µJ | ETTm2 H96 MSE / MAE / energy µJ |
|---|---|---|
| Spikformer-style | 0.2903 / 0.3350 / 9.30 | 0.1767 / 0.2533 / 10.29 |
| Spikformer-style + H1 | 0.2929 / 0.3393 / 13.99 | 0.1773 / 0.2543 / 14.74 |
| muxlin | 0.2952 / 0.3381 / 9.06 | 0.1775 / 0.2547 / 9.89 |
| muxlin + H1 | 0.3081 / 0.3500 / 13.72 | 0.1797 / 0.2571 / 14.33 |
