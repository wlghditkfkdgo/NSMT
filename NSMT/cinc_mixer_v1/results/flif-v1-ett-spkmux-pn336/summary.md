# Part 4 (reduced by the user decision of 11-18 21:4x), ETT input 336: LIF SpkMixer vs SF-de

Rules and sources: module docstring of summarize_pn.py. Seeds 42 / 7 / 2026.

## Test MSE (seed mean) and energy

| arm | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean | mean best validation L1 | energy µJ (4-cell mean) | params (ETTh2 H96) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| LIF 336 | 0.2964 | 0.3916 | 0.1726 | 0.3778 | 0.3096 | 0.36450 | 34.69 | 537920 |
| pop LIF all 336 | 0.2976 | 0.4202 | 0.1696 | 0.3631 | 0.3126 | 0.36252 | 98.68 | 539584 |
| SF-sj 336 | 0.3094 | 0.3983 | 0.1715 | 0.3673 | 0.3116 | 0.36412 | 35.13 | 539256 |
| SF-de 336 | 0.3044 | 0.3952 | 0.1721 | 0.3697 | 0.3103 | 0.36409 | 35.61 | 539256 |

## Comparisons

| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % | validation L1 R % |
|---|--:|--:|---|--:|---|--:|--:|
| D1 LIF 336 vs SF-de 336 | -0.25 | 7/12 | no clear difference | 0.97 | -2.64, -0.92, +0.35, +2.19 | -0.38 | -0.00 |
| (report) pop LIF all 336 vs SF-de 336 | +0.23 | 8/12 | no clear difference | 2.78 | -2.23, +6.33, -1.40, -1.76 | +0.03 | -0.52 |
| (report) SF-de 336 vs SF-sj 336 | -0.33 | 8/12 | no clear difference | 1.01 | -1.52, -0.78, +0.32, +0.66 | +0.04 | +0.05 |
| (report) LIF 336 vs SF-sj 336 (= 11-18 P2) | -0.57 | 6/12 | no clear difference | 0.99 | -4.08, -1.69, +0.62, +2.86 | -0.34 | +0.05 |

Pairs r % (test MSE), seed 42, 7, 2026 per cell (verdict comparisons):

- D1 LIF 336 vs SF-de 336: ETTh2 H96 -1.00, -3.25, -3.66; ETTh2 H720 -0.35, -1.28, -1.12; ETTm2 H96 -2.34, +2.12, +1.25; ETTm2 H720 +2.80, +2.54, +1.23

## Report only: stopping and firing rates (seed mean)

| cell | arm | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | energy µJ | time s (seeds) |
|---|---|---|---|--:|--:|--:|---|
| ETTh2 H96 | LIF 336 | 1, 1, 1 | 12, 12, 12 | 0.3092 | 0.0736 | 31.38 | 1481, 1520, 1453 |
| ETTh2 H96 | pop LIF all 336 | 2, 1, 0 | 13, 12, 11 | 0.3239 | 0.0779 | 92.94 | 3772, 4011, 3556 |
| ETTh2 H96 | SF-sj 336 | 1, 4, 2 | 12, 15, 13 | 0.3150 | 0.0549 | 32.26 | 198, 241, 216 |
| ETTh2 H96 | SF-de 336 | 1, 2, 0 | 12, 13, 11 | 0.3186 | 0.0591 | 32.25 | 398, 395, 343 |
| ETTh2 H720 | LIF 336 | 0, 0, 0 | 11, 11, 11 | 0.4019 | 0.0732 | 34.78 | 1336, 1090, 1154 |
| ETTh2 H720 | pop LIF all 336 | 3, 7, 3 | 14, 18, 14 | 0.4277 | 0.0843 | 96.98 | 4746, 4723, 4785 |
| ETTh2 H720 | SF-sj 336 | 0, 0, 0 | 11, 11, 11 | 0.4113 | 0.0461 | 35.02 | 185, 161, 189 |
| ETTh2 H720 | SF-de 336 | 0, 0, 0 | 11, 11, 11 | 0.4057 | 0.0548 | 35.73 | 360, 331, 331 |
| ETTm2 H96 | LIF 336 | 7, 13, 12 | 18, 24, 23 | 0.1731 | 0.0804 | 34.37 | 8346, 11360, 11419 |
| ETTm2 H96 | pop LIF all 336 | 3, 3, 2 | 14, 14, 13 | 0.1737 | 0.0693 | 99.74 | 26194, 20408, 18344 |
| ETTm2 H96 | SF-sj 336 | 2, 3, 2 | 13, 14, 13 | 0.1758 | 0.0588 | 35.30 | 942, 948, 947 |
| ETTm2 H96 | SF-de 336 | 7, 3, 9 | 18, 14, 20 | 0.1733 | 0.0626 | 35.21 | 1962, 1694, 1921 |
| ETTm2 H720 | LIF 336 | 1, 2, 0 | 12, 13, 11 | 0.3808 | 0.0698 | 38.21 | 5140, 6447, 5528 |
| ETTm2 H720 | pop LIF all 336 | 0, 0, 0 | 11, 11, 11 | 0.3851 | 0.0646 | 105.05 | 17819, 17147, 22989 |
| ETTm2 H720 | SF-sj 336 | 0, 0, 0 | 11, 11, 11 | 0.3790 | 0.0460 | 37.94 | 943, 782, 813 |
| ETTm2 H720 | SF-de 336 | 0, 0, 0 | 11, 11, 11 | 0.3839 | 0.0511 | 39.26 | 1518, 1318, 1492 |

## Paper bar (user decision 11-18): SpkMixer mean better than SF-de and less energy, sign only

| comparison | R % | mean better (R < 0) | energy A/B | less energy (< 1) | paper bar |
|---|--:|---|--:|---|---|
| D1 LIF 336 vs SF-de 336 | -0.254 | yes | 0.9740 | yes | met |
| (report) pop LIF all 336 vs SF-de 336 | +0.234 | no | 2.7762 | no | not met |
