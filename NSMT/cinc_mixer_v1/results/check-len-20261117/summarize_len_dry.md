<!-- dry run of summarize_len.py (2026-11-17): synthetic input-336 rows = the input-96 rows with the tag renamed; every P3 R and every G must be 0 and P1 / P2 must equal the input-96 rows. recheck_len.py --dir on the same rows gives the same values. Not a result. -->
# ETT input-length test on the SpkMixer, input 336 against input 96

Rules and sources: module docstring of summarize_len.py. Seeds 42 / 7 / 2026. Tuned input-96 settings, not re-tuned.

## Test MSE (seed mean) and energy

| model | input | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean | mean best validation L1 | energy µJ (4-cell mean) | params (ETTh2 H96) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| LIF SpkMixer | 96 | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| LIF SpkMixer | 336 | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| population LIF SpkMixer | 96 | 0.2910 | 0.4134 | 0.1769 | 0.4033 | 0.3212 | 0.36151 | 28.50 | 170944 |
| population LIF SpkMixer | 336 | 0.2910 | 0.4134 | 0.1769 | 0.4033 | 0.3212 | 0.36151 | 28.50 | 170944 |
| Spikformer-style | 96 | 0.2903 | 0.4223 | 0.1767 | 0.4044 | 0.3234 | 0.36087 | 10.48 | 169656 |
| Spikformer-style | 336 | 0.2903 | 0.4223 | 0.1767 | 0.4044 | 0.3234 | 0.36087 | 10.48 | 169656 |

## Pre-registered comparisons

| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % |
|---|--:|--:|---|--:|---|--:|
| P1 POP 336 vs LIF 336 | -0.94 | 10/12 | no clear difference | 2.78 | -1.41, -1.98, -0.34, -0.02 | -0.71 |
| P2 POP 336 vs SF 336 | -0.50 | 8/12 | no clear difference | 2.73 | +0.23, -2.11, +0.13, -0.26 | -0.27 |
| P2 LIF 336 vs SF 336 | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 |
| P3 LIF 336 vs LIF 96 | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 |
| P3 POP 336 vs POP 96 | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 |
| P3 SF 336 vs SF 96 | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 |
| (input 96) POP vs LIF | -0.94 | 10/12 | no clear difference | 2.78 | -1.41, -1.98, -0.34, -0.02 | -0.71 |
| (input 96) POP vs SF | -0.50 | 8/12 | no clear difference | 2.73 | +0.23, -2.11, +0.13, -0.26 | -0.27 |
| (input 96) LIF vs SF | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 |

## Gap change from input 96 (g = r(L) - r(96), %p; negative = the first model gains more from the longer input)

| pair | G %p | negative pairs | verdict | per cell g %p |
|---|--:|--:|---|---|
| POP vs LIF | +0.00 | 0/12 | no change | +0.00, +0.00, +0.00, +0.00 |
| POP vs SF | +0.00 | 0/12 | no change | +0.00, +0.00, +0.00, +0.00 |
| LIF vs SF | +0.00 | 0/12 | no change | +0.00, +0.00, +0.00, +0.00 |

Pairs r % (test MSE), seed 42, 7, 2026 per cell:

- P1 POP 336 vs LIF 336: ETTh2 H96 -3.23, +0.12, -1.11; ETTh2 H720 -3.41, -1.25, -1.28; ETTm2 H96 -0.31, -0.01, -0.70; ETTm2 H720 +0.36, -0.15, -0.29
- P2 POP 336 vs SF 336: ETTh2 H96 -0.93, +0.51, +1.12; ETTh2 H720 -2.61, -1.79, -1.92; ETTm2 H96 -0.12, +0.42, +0.07; ETTm2 H720 -0.53, -0.07, -0.17
- P2 LIF 336 vs SF 336: ETTh2 H96 +2.37, +0.39, +2.25; ETTh2 H720 +0.83, -0.55, -0.64; ETTm2 H96 +0.19, +0.43, +0.77; ETTm2 H720 -0.89, +0.07, +0.12
- P3 LIF 336 vs LIF 96: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00
- P3 POP 336 vs POP 96: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00
- P3 SF 336 vs SF 96: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00

## Report only: the same comparisons on best validation L1 and on last-epoch test MSE

| comparison | validation L1 R % | lower pairs | last-epoch MSE R % | lower pairs |
|---|--:|--:|--:|--:|
| P1 POP 336 vs LIF 336 | +0.01 | 6/12 | -0.54 | 9/12 |
| P2 POP 336 vs SF 336 | +0.18 | 3/12 | -0.26 | 7/12 |
| P2 LIF 336 vs SF 336 | +0.17 | 3/12 | +0.29 | 7/12 |
| P3 LIF 336 vs LIF 96 | +0.00 | 0/12 | +0.00 | 0/12 |
| P3 POP 336 vs POP 96 | +0.00 | 0/12 | +0.00 | 0/12 |
| P3 SF 336 vs SF 96 | +0.00 | 0/12 | +0.00 | 0/12 |

## Report only: stopping, firing rate, energy per cell (seed mean)

| cell | model | input | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | energy µJ | params |
|---|---|--:|---|---|--:|--:|--:|--:|
| ETTh2 H96 | LIF | 96 | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 | 169280 |
| ETTh2 H96 | LIF | 336 | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 | 169280 |
| ETTh2 H96 | POP | 96 | 0, 1, 1 | 11, 12, 12 | 0.2892 | 0.0715 | 26.63 | 170944 |
| ETTh2 H96 | POP | 336 | 0, 1, 1 | 11, 12, 12 | 0.2892 | 0.0715 | 26.63 | 170944 |
| ETTh2 H96 | SF | 96 | 3, 1, 7 | 14, 12, 18 | 0.2879 | 0.0505 | 9.30 | 169656 |
| ETTh2 H96 | SF | 336 | 3, 1, 7 | 14, 12, 18 | 0.2879 | 0.0505 | 9.30 | 169656 |
| ETTh2 H720 | LIF | 96 | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 | 1127744 |
| ETTh2 H720 | LIF | 336 | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 | 1127744 |
| ETTh2 H720 | POP | 96 | 1, 0, 2 | 12, 11, 13 | 0.4201 | 0.0678 | 28.32 | 1129408 |
| ETTh2 H720 | POP | 336 | 1, 0, 2 | 12, 11, 13 | 0.4201 | 0.0678 | 28.32 | 1129408 |
| ETTh2 H720 | SF | 96 | 3, 3, 4 | 14, 14, 15 | 0.4223 | 0.0493 | 10.79 | 1128120 |
| ETTh2 H720 | SF | 336 | 3, 3, 4 | 14, 14, 15 | 0.4223 | 0.0493 | 10.79 | 1128120 |
| ETTm2 H96 | LIF | 96 | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 | 173440 |
| ETTm2 H96 | LIF | 336 | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 | 173440 |
| ETTm2 H96 | POP | 96 | 8, 5, 6 | 19, 16, 17 | 0.1768 | 0.0799 | 28.62 | 175232 |
| ETTm2 H96 | POP | 336 | 8, 5, 6 | 19, 16, 17 | 0.1768 | 0.0799 | 28.62 | 175232 |
| ETTm2 H96 | SF | 96 | 11, 11, 4 | 22, 22, 15 | 0.1770 | 0.0695 | 10.29 | 174192 |
| ETTm2 H96 | SF | 336 | 11, 11, 4 | 22, 22, 15 | 0.1770 | 0.0695 | 10.29 | 174192 |
| ETTm2 H720 | LIF | 96 | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 | 1131904 |
| ETTm2 H720 | LIF | 336 | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 | 1131904 |
| ETTm2 H720 | POP | 96 | 2, 1, 1 | 13, 12, 12 | 0.4104 | 0.0705 | 30.43 | 1133696 |
| ETTm2 H720 | POP | 336 | 2, 1, 1 | 13, 12, 12 | 0.4104 | 0.0705 | 30.43 | 1133696 |
| ETTm2 H720 | SF | 96 | 2, 1, 1 | 13, 12, 12 | 0.4139 | 0.0539 | 11.54 | 1132656 |
| ETTm2 H720 | SF | 336 | 2, 1, 1 | 13, 12, 12 | 0.4139 | 0.0539 | 11.54 | 1132656 |
