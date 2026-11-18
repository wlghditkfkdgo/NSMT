# ETT input-length test on the SpkMixer, input 336 against input 96

Rules and sources: module docstring of summarize_len.py. Seeds 42 / 7 / 2026. Tuned input-96 settings, not re-tuned.

## Test MSE (seed mean) and energy

| model | input | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean | mean best validation L1 | energy µJ (4-cell mean) | params (ETTh2 H96) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| LIF SpkMixer | 96 | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| LIF SpkMixer | 336 | 0.2964 | 0.3916 | 0.1726 | 0.3778 | 0.3096 | 0.36450 | 34.69 | 537920 |
| population LIF SpkMixer | 96 | 0.2910 | 0.4134 | 0.1769 | 0.4033 | 0.3212 | 0.36151 | 28.50 | 170944 |
| population LIF SpkMixer | 336 | 0.2976 | 0.4202 | 0.1696 | 0.3631 | 0.3126 | 0.36252 | 98.68 | 539584 |
| Spikformer-style | 96 | 0.2903 | 0.4223 | 0.1767 | 0.4044 | 0.3234 | 0.36087 | 10.48 | 169656 |
| Spikformer-style | 336 | 0.3094 | 0.3983 | 0.1715 | 0.3673 | 0.3116 | 0.36412 | 35.13 | 539256 |

## Pre-registered comparisons

| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % |
|---|--:|--:|---|--:|---|--:|
| P1 POP 336 vs LIF 336 | +0.53 | 7/12 | no clear difference | 2.85 | +0.42, +7.31, -1.72, -3.86 | +0.41 |
| P2 POP 336 vs SF 336 | -0.10 | 7/12 | no clear difference | 2.81 | -3.68, +5.50, -1.11, -1.12 | +0.06 |
| P2 LIF 336 vs SF 336 | -0.57 | 6/12 | no clear difference | 0.99 | -4.08, -1.69, +0.62, +2.86 | -0.34 |
| P3 LIF 336 vs LIF 96 | -3.98 | 10/12 | better | 3.38 | +0.39, -7.17, -2.78, -6.36 | -0.94 |
| P3 POP 336 vs POP 96 | -2.55 | 7/12 | no clear difference | 3.46 | +2.27, +1.63, -4.14, -9.96 | +0.17 |
| P3 SF 336 vs SF 96 | -2.80 | 9/12 | better | 3.36 | +6.59, -5.69, -2.92, -9.18 | -0.11 |
| (input 96) POP vs LIF | -0.94 | 10/12 | no clear difference | 2.78 | -1.41, -1.98, -0.34, -0.02 | -0.71 |
| (input 96) POP vs SF | -0.50 | 8/12 | no clear difference | 2.73 | +0.23, -2.11, +0.13, -0.26 | -0.27 |
| (input 96) LIF vs SF | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 |

## Gap change from input 96 (g = r(L) - r(96), %p; negative = the first model gains more from the longer input)

| pair | G %p | negative pairs | verdict | per cell g %p |
|---|--:|--:|---|---|
| POP vs LIF | +1.47 | 5/12 | no change | +1.82, +9.29, -1.38, -3.84 |
| POP vs SF | +0.40 | 7/12 | no change | -3.91, +7.61, -1.23, -0.86 |
| LIF vs SF | -1.02 | 7/12 | no change | -5.75, -1.57, +0.15, +3.09 |

Pairs r % (test MSE), seed 42, 7, 2026 per cell:

- P1 POP 336 vs LIF 336: ETTh2 H96 +1.03, +1.23, -1.01; ETTh2 H720 +8.34, +5.23, +8.34; ETTm2 H96 -0.24, -2.11, -2.82; ETTm2 H720 -5.28, -3.35, -2.96
- P2 POP 336 vs SF 336: ETTh2 H96 +2.89, -7.81, -6.12; ETTh2 H720 +7.26, +2.74, +6.50; ETTm2 H96 +1.14, -2.54, -1.92; ETTm2 H720 -2.02, -1.12, -0.23
- P2 LIF 336 vs SF 336: ETTh2 H96 +1.84, -8.92, -5.16; ETTh2 H720 -1.00, -2.37, -1.69; ETTm2 H96 +1.38, -0.44, +0.92; ETTm2 H720 +3.44, +2.31, +2.82
- P3 LIF 336 vs LIF 96: ETTh2 H96 +0.70, -0.14, +0.61; ETTh2 H720 -7.67, -6.48, -7.36; ETTm2 H96 -3.21, -2.92, -2.21; ETTm2 H720 -5.57, -6.99, -6.54
- P3 POP 336 vs POP 96: ETTh2 H96 +5.13, +0.96, +0.71; ETTh2 H720 +3.57, -0.34, +1.67; ETTm2 H96 -3.15, -4.96, -4.30; ETTm2 H720 -10.88, -9.97, -9.05
- P3 SF 336 vs SF 96: ETTh2 H96 +1.22, +10.07, +8.47; ETTh2 H720 -5.97, -4.73, -6.37; ETTm2 H96 -4.35, -2.07, -2.35; ETTm2 H720 -9.52, -9.02, -9.00

## Report only: the same comparisons on best validation L1 and on last-epoch test MSE

| comparison | validation L1 R % | lower pairs | last-epoch MSE R % | lower pairs |
|---|--:|--:|--:|--:|
| P1 POP 336 vs LIF 336 | -0.52 | 11/12 | +3.16 | 2/12 |
| P2 POP 336 vs SF 336 | -0.47 | 10/12 | +1.82 | 3/12 |
| P2 LIF 336 vs SF 336 | +0.05 | 6/12 | -1.29 | 10/12 |
| P3 LIF 336 vs LIF 96 | +0.38 | 6/12 | -2.26 | 9/12 |
| P3 POP 336 vs POP 96 | -0.16 | 6/12 | +1.48 | 6/12 |
| P3 SF 336 vs SF 96 | +0.49 | 6/12 | -0.58 | 8/12 |

## Report only: stopping, firing rate, energy per cell (seed mean)

| cell | model | input | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | energy µJ | params |
|---|---|--:|---|---|--:|--:|--:|--:|
| ETTh2 H96 | LIF | 96 | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 | 169280 |
| ETTh2 H96 | LIF | 336 | 1, 1, 1 | 12, 12, 12 | 0.3092 | 0.0736 | 31.38 | 537920 |
| ETTh2 H96 | POP | 96 | 0, 1, 1 | 11, 12, 12 | 0.2892 | 0.0715 | 26.63 | 170944 |
| ETTh2 H96 | POP | 336 | 2, 1, 0 | 13, 12, 11 | 0.3239 | 0.0779 | 92.94 | 539584 |
| ETTh2 H96 | SF | 96 | 3, 1, 7 | 14, 12, 18 | 0.2879 | 0.0505 | 9.30 | 169656 |
| ETTh2 H96 | SF | 336 | 1, 4, 2 | 12, 15, 13 | 0.3150 | 0.0549 | 32.26 | 539256 |
| ETTh2 H720 | LIF | 96 | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 | 1127744 |
| ETTh2 H720 | LIF | 336 | 0, 0, 0 | 11, 11, 11 | 0.4019 | 0.0732 | 34.78 | 3892544 |
| ETTh2 H720 | POP | 96 | 1, 0, 2 | 12, 11, 13 | 0.4201 | 0.0678 | 28.32 | 1129408 |
| ETTh2 H720 | POP | 336 | 3, 7, 3 | 14, 18, 14 | 0.4277 | 0.0843 | 96.98 | 3894208 |
| ETTh2 H720 | SF | 96 | 3, 3, 4 | 14, 14, 15 | 0.4223 | 0.0493 | 10.79 | 1128120 |
| ETTh2 H720 | SF | 336 | 0, 0, 0 | 11, 11, 11 | 0.4113 | 0.0461 | 35.02 | 3893880 |
| ETTm2 H96 | LIF | 96 | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 | 173440 |
| ETTm2 H96 | LIF | 336 | 7, 13, 12 | 18, 24, 23 | 0.1731 | 0.0804 | 34.37 | 542080 |
| ETTm2 H96 | POP | 96 | 8, 5, 6 | 19, 16, 17 | 0.1768 | 0.0799 | 28.62 | 175232 |
| ETTm2 H96 | POP | 336 | 3, 3, 2 | 14, 14, 13 | 0.1737 | 0.0693 | 99.74 | 543872 |
| ETTm2 H96 | SF | 96 | 11, 11, 4 | 22, 22, 15 | 0.1770 | 0.0695 | 10.29 | 174192 |
| ETTm2 H96 | SF | 336 | 2, 3, 2 | 13, 14, 13 | 0.1758 | 0.0588 | 35.30 | 544752 |
| ETTm2 H720 | LIF | 96 | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 | 1131904 |
| ETTm2 H720 | LIF | 336 | 1, 2, 0 | 12, 13, 11 | 0.3808 | 0.0698 | 38.21 | 3896704 |
| ETTm2 H720 | POP | 96 | 2, 1, 1 | 13, 12, 12 | 0.4104 | 0.0705 | 30.43 | 1133696 |
| ETTm2 H720 | POP | 336 | 0, 0, 0 | 11, 11, 11 | 0.3851 | 0.0646 | 105.05 | 3898496 |
| ETTm2 H720 | SF | 96 | 2, 1, 1 | 13, 12, 12 | 0.4139 | 0.0539 | 11.54 | 1132656 |
| ETTm2 H720 | SF | 336 | 0, 0, 0 | 11, 11, 11 | 0.3790 | 0.0460 | 37.94 | 3899376 |
