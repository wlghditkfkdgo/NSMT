# Part 1, ETT: LIF SpkMixer vs the Spikformer-style model with the same spikeDE LIF (SF-de)

Rules and sources: module docstring of summarize_pn.py. Seeds 42 / 7 / 2026.

## Test MSE (seed mean) and energy

| arm | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean | mean best validation L1 | energy µJ (4-cell mean) | params (ETTh2 H96) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| LIF SpkMixer | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| SF-de | 0.2921 | 0.4229 | 0.1768 | 0.4036 | 0.3238 | 0.36126 | 10.67 | 169656 |
| SF-sj | 0.2903 | 0.4223 | 0.1767 | 0.4044 | 0.3234 | 0.36087 | 10.48 | 169656 |

## Comparisons

| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % | validation L1 R % |
|---|--:|--:|---|--:|---|--:|--:|
| A1 LIF SpkMixer vs SF-de | +0.30 | 5/12 | no clear difference | 0.96 | +1.08, -0.24, +0.42, -0.04 | +0.30 | +0.05 |
| (report) SF-de vs SF-sj | +0.15 | 7/12 | no clear difference | 1.02 | +0.61, +0.13, +0.04, -0.19 | +0.15 | +0.12 |
| (report) LIF SpkMixer vs SF-sj | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |

Pairs r % (test MSE), seed 42, 7, 2026 per cell (verdict comparisons):

- A1 LIF SpkMixer vs SF-de: ETTh2 H96 +3.57, -1.48, +1.16; ETTh2 H720 -0.78, +0.32, -0.28; ETTm2 H96 +0.34, +0.10, +0.83; ETTm2 H720 -0.32, -0.16, +0.35

## Report only: stopping and firing rates (seed mean)

| cell | arm | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | energy µJ | time s (seeds) |
|---|---|---|---|--:|--:|--:|---|
| ETTh2 H96 | LIF SpkMixer | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 | 461, 404, 498 |
| ETTh2 H96 | SF-de | 1, 1, 3 | 12, 12, 14 | 0.2894 | 0.0572 | 9.37 | 280, 253, 302 |
| ETTh2 H96 | SF-sj | 3, 1, 7 | 14, 12, 18 | 0.2879 | 0.0505 | 9.30 | 263, 235, 234 |
| ETTh2 H720 | LIF SpkMixer | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 | 377, 342, 416 |
| ETTh2 H720 | SF-de | 3, 3, 2 | 14, 14, 13 | 0.4233 | 0.0534 | 10.96 | 301, 309, 329 |
| ETTh2 H720 | SF-sj | 3, 3, 4 | 14, 14, 15 | 0.4223 | 0.0493 | 10.79 | 360, 363, 359 |
| ETTm2 H96 | LIF SpkMixer | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 | 3829, 2399, 2162 |
| ETTm2 H96 | SF-de | 15, 11, 12 | 26, 22, 23 | 0.1769 | 0.0727 | 10.34 | 1513, 1276, 1326 |
| ETTm2 H96 | SF-sj | 11, 11, 4 | 22, 22, 15 | 0.1770 | 0.0695 | 10.29 | 606, 578, 692 |
| ETTm2 H720 | LIF SpkMixer | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 | 1729, 1905, 1600 |
| ETTm2 H720 | SF-de | 2, 4, 2 | 13, 15, 13 | 0.4091 | 0.0627 | 11.99 | 1051, 1132, 992 |
| ETTm2 H720 | SF-sj | 2, 1, 1 | 13, 12, 12 | 0.4139 | 0.0539 | 11.54 | 428, 437, 344 |
