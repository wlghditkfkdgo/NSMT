# MSE-loss test of the tuned SpkMixer on ETT

Rules and sources: module docstring. A = MSE loss, B = L1 loss; same model, lr, seeds, patience. Seeds 42 / 7 / 2026.

## Test values (seed mean)

| cell | MSE, L1 loss | MSE, MSE loss | r % | MAE, L1 loss | MAE, MSE loss | r % | energy µJ, L1 / MSE |
|---|---:|---:|---:|---:|---:|---:|---|
| ETTh2 H96 | 0.2952 | 0.3005 | +1.82 | 0.3381 | 0.3472 | +2.69 | 9.06 / 8.98 |
| ETTh2 H720 | 0.4218 | 0.4339 | +2.85 | 0.4384 | 0.4505 | +2.76 | 10.67 / 10.59 |
| ETTm2 H96 | 0.1775 | 0.1803 | +1.53 | 0.2547 | 0.2649 | +3.98 | 9.89 / 9.61 |
| ETTm2 H720 | 0.4034 | 0.4044 | +0.23 | 0.3955 | 0.4023 | +1.72 | 11.49 / 11.04 |

## Pre-registered comparison: MSE loss vs L1 loss

| metric | R % | lower pairs | verdict | energy A/B | per cell r % |
|---|---:|---:|---|---:|---|
| test MSE (verdict) | +1.61 | 1/12 | worse | 0.98 | +1.82, +2.85, +1.53, +0.23 |
| test MAE (same rule, report) | +2.78 | 0/12 | worse | 0.98 | +2.69, +2.76, +3.98, +1.72 |

Pairs r % (test MSE), seed 42, 7, 2026 per cell:

- ETTh2 H96: +0.48, +4.03, +0.95
- ETTh2 H720: +3.51, +2.59, +2.46
- ETTm2 H96: +1.44, +1.46, +1.70
- ETTm2 H720: +0.03, +1.00, -0.35

## Report only: stopping and validation

| cell | best epoch, L1 loss (seeds) | best epoch, MSE loss (seeds) | epochs run, L1 / MSE (seeds) | validation MSE: L1 checkpoint | L1 lowest over its epochs | MSE checkpoint |
|---|---|---|---|---:|---:|---:|
| ETTh2 H96 | 3, 1, 6 | 0, 1, 0 | 14/12/17 ; 11/12/11 | 0.2168 | 0.2161 | 0.2155 |
| ETTh2 H720 | 1, 0, 2 | 1, 0, 2 | 12/11/13 ; 12/11/13 | 0.6138 | 0.6138 | 0.5917 |
| ETTm2 H96 | 25, 11, 7 | 4, 6, 7 | 36/22/18 ; 15/17/18 | 0.1228 | 0.1226 | 0.1257 |
| ETTm2 H720 | 2, 4, 2 | 2, 5, 2 | 13/15/13 ; 13/16/13 | 0.2863 | 0.2863 | 0.2860 |

## Report only: firing rates and last-epoch weights (seed mean)

| cell | rho_neo, L1 / MSE | rho_hippo, L1 / MSE | last-epoch test MSE, L1 / MSE |
|---|---|---|---|
| ETTh2 H96 | 0.0463 / 0.0446 | 0.0704 / 0.0657 | 0.2960 / 0.2988 |
| ETTh2 H720 | 0.0427 / 0.0423 | 0.0629 / 0.0624 | 0.4191 / 0.4234 |
| ETTm2 H96 | 0.0468 / 0.0416 | 0.0832 / 0.0726 | 0.1773 / 0.1808 |
| ETTm2 H720 | 0.0515 / 0.0396 | 0.0744 / 0.0688 | 0.4096 / 0.4157 |

## Report only: against tuned Spikformer-style (trained with L1 only; the MSE-loss row mixes losses)

| comparison | metric | R % | lower pairs | verdict | energy A/B | per cell r % |
|---|---|---:|---:|---|---:|---|
| SpkMixer L1 loss vs Spikformer-style L1 loss | MSE | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 |
| SpkMixer L1 loss vs Spikformer-style L1 loss | MAE | +0.45 | 2/12 | no clear difference | 0.98 | +0.95, +0.41, +0.55, -0.13 |
| SpkMixer MSE loss vs Spikformer-style L1 loss | MSE | +2.06 | 2/12 | worse | 0.96 | +3.51, +2.73, +2.01, -0.01 |
| SpkMixer MSE loss vs Spikformer-style L1 loss | MAE | +3.24 | 0/12 | worse | 0.96 | +3.66, +3.18, +4.55, +1.58 |
