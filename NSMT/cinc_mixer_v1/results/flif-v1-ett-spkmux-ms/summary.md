# Membrane shortcut test, ETT (LIF only)

Rules and sources: module docstring of summarize_ms.py. Seeds 42 / 7 / 2026. Tuned settings, not re-tuned for the shortcut.

## Test values (seed mean)

| cell | SpkMixer | SpkMixer + MS | Spikformer-style | Spikformer-style + MS |
|---|--:|--:|--:|--:|
| ETTh2 H96 | 0.2952 | 0.2934 | 0.2903 | 0.2903 |
| ETTh2 H720 | 0.4218 | 0.4175 | 0.4223 | 0.4190 |
| ETTm2 H96 | 0.1775 | 0.1771 | 0.1767 | 0.1761 |
| ETTm2 H720 | 0.4034 | 0.4050 | 0.4044 | 0.4032 |

## Pre-registered comparisons

| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % |
|---|--:|--:|---|--:|---|--:|
| M1 SpkMixer + MS vs SpkMixer | -0.36 | 8/12 | no clear difference | 1.21 | -0.58, -1.01, -0.24, +0.39 | -0.34 |
| M2 Spikformer-style + MS vs Spikformer-style | -0.35 | 10/12 | no clear difference | 1.14 | -0.01, -0.78, -0.33, -0.29 | -0.26 |
| M3 SpkMixer + MS vs Spikformer-style + MS | +0.44 | 2/12 | no clear difference | 1.04 | +1.09, -0.36, +0.56, +0.45 | +0.36 |
| M4 (report only) SpkMixer + MS vs Spikformer-style | +0.08 | 4/12 | no clear difference | 1.19 | +1.07, -1.13, +0.23, +0.16 | +0.10 |
| (reference) SpkMixer vs Spikformer-style | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 |

Pairs r % (test MSE), seed 42, 7, 2026 per cell:

- M1 SpkMixer + MS vs SpkMixer: ETTh2 H96 -2.23, +1.34, -0.86; ETTh2 H720 -0.53, -0.13, -2.37; ETTm2 H96 -0.31, +0.15, -0.56; ETTm2 H720 -0.23, +0.36, +1.04
- M2 Spikformer-style + MS vs Spikformer-style: ETTh2 H96 -1.53, +1.62, -0.13; ETTh2 H720 -0.25, -0.62, -1.47; ETTm2 H96 -0.78, -0.15, -0.07; ETTm2 H720 -1.47, -0.05, +0.65
- M3 SpkMixer + MS vs Spikformer-style + MS: ETTh2 H96 +1.65, +0.11, +1.51; ETTh2 H720 +0.54, -0.06, -1.55; ETTm2 H96 +0.67, +0.73, +0.28; ETTm2 H720 +0.36, +0.48, +0.51

## Report only: stopping, last-epoch weights, firing rates (seed mean)

| cell | model | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | rho_neo | energy µJ |
|---|---|---|---|--:|--:|--:|--:|
| ETTh2 H96 | SpkMixer | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 0.0463 | 9.06 |
| ETTh2 H96 | SpkMixer + MS | 0, 1, 0 | 11, 12, 11 | 0.3117 | 0.0892 | 0.0494 | 10.36 |
| ETTh2 H96 | Spikformer-style | 3, 1, 7 | 14, 12, 18 | 0.2879 | 0.0505 | - | 9.30 |
| ETTh2 H96 | Spikformer-style + MS | 0, 1, 0 | 11, 12, 11 | 0.2931 | 0.0559 | - | 10.36 |
| ETTh2 H720 | SpkMixer | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 0.0427 | 10.67 |
| ETTh2 H720 | SpkMixer + MS | 0, 0, 0 | 11, 11, 11 | 0.4468 | 0.0848 | 0.0454 | 13.41 |
| ETTh2 H720 | Spikformer-style | 3, 3, 4 | 14, 14, 15 | 0.4223 | 0.0493 | - | 10.79 |
| ETTh2 H720 | Spikformer-style + MS | 0, 0, 2 | 11, 11, 13 | 0.4266 | 0.0533 | - | 12.79 |
| ETTm2 H96 | SpkMixer | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 0.0468 | 9.89 |
| ETTm2 H96 | SpkMixer + MS | 3, 3, 4 | 14, 14, 15 | 0.1791 | 0.0923 | 0.0580 | 11.16 |
| ETTm2 H96 | Spikformer-style | 11, 11, 4 | 22, 22, 15 | 0.1770 | 0.0695 | - | 10.29 |
| ETTm2 H96 | Spikformer-style + MS | 3, 5, 2 | 14, 16, 13 | 0.1776 | 0.0613 | - | 10.98 |
| ETTm2 H720 | SpkMixer | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 0.0515 | 11.49 |
| ETTm2 H720 | SpkMixer + MS | 0, 0, 1 | 11, 11, 12 | 0.4203 | 0.0921 | 0.0552 | 15.01 |
| ETTm2 H720 | Spikformer-style | 2, 1, 1 | 13, 12, 12 | 0.4139 | 0.0539 | - | 11.54 |
| ETTm2 H720 | Spikformer-style + MS | 0, 0, 0 | 11, 11, 11 | 0.4206 | 0.0534 | - | 13.55 |
