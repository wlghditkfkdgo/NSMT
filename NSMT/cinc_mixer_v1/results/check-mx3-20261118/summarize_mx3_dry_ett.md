# Mixer variants of the LIF SpkMixer, ETT

Rules and sources: module docstring of summarize_mx3.py. Seeds 42 / 7 / 2026. Tuned LIF SpkMixer settings, not re-tuned.

## Test MSE (seed mean) and energy

| arm | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean | mean best validation L1 | energy µJ (4-cell mean) | params (ETTh2 H96) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| base | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| max | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| mean | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| L2 | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| MS | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| subtract | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| zero | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| Spikformer-style | 0.2903 | 0.4223 | 0.1767 | 0.4044 | 0.3234 | 0.36087 | 10.48 | 169656 |

## Pre-registered comparisons

| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % | validation L1 R % |
|---|--:|--:|---|--:|---|--:|--:|
| V1 max vs base | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| V2 mean vs base | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| V3 MS vs base | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| V4 MS vs L2 | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| V5 subtract vs base | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| V6 zero vs base | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| (report) max vs mean | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| (report) L2 vs base | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| (report) subtract vs zero | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| (report) base vs Spikformer-style | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) max vs Spikformer-style | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) mean vs Spikformer-style | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) L2 vs Spikformer-style | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) MS vs Spikformer-style | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) subtract vs Spikformer-style | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) zero vs Spikformer-style | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |

Pairs r % (test MSE), seed 42, 7, 2026 per cell:

- V1 max vs base: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00
- V2 mean vs base: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00
- V3 MS vs base: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00
- V4 MS vs L2: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00
- V5 subtract vs base: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00
- V6 zero vs base: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00

## Report only: stopping and firing rates (seed mean)

| cell | arm | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | energy µJ |
|---|---|---|---|--:|--:|--:|
| ETTh2 H96 | base | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 |
| ETTh2 H96 | max | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 |
| ETTh2 H96 | mean | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 |
| ETTh2 H96 | L2 | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 |
| ETTh2 H96 | MS | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 |
| ETTh2 H96 | subtract | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 |
| ETTh2 H96 | zero | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 |
| ETTh2 H720 | base | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 |
| ETTh2 H720 | max | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 |
| ETTh2 H720 | mean | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 |
| ETTh2 H720 | L2 | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 |
| ETTh2 H720 | MS | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 |
| ETTh2 H720 | subtract | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 |
| ETTh2 H720 | zero | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 |
| ETTm2 H96 | base | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 |
| ETTm2 H96 | max | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 |
| ETTm2 H96 | mean | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 |
| ETTm2 H96 | L2 | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 |
| ETTm2 H96 | MS | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 |
| ETTm2 H96 | subtract | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 |
| ETTm2 H96 | zero | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 |
| ETTm2 H720 | base | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 |
| ETTm2 H720 | max | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 |
| ETTm2 H720 | mean | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 |
| ETTm2 H720 | L2 | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 |
| ETTm2 H720 | MS | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 |
| ETTm2 H720 | subtract | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 |
| ETTm2 H720 | zero | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 |
