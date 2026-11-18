# Parts 2 and 3, ETT: SpkMixer neuron x placement x mean pooling

Rules and sources: module docstring of summarize_pn.py. Seeds 42 / 7 / 2026.

## Test MSE (seed mean) and energy

| arm | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean | mean best validation L1 | energy µJ (4-cell mean) | params (ETTh2 H96) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| LIF | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| LIF+mean | 0.2893 | 0.4199 | 0.1773 | 0.4033 | 0.3225 | 0.36242 | 13.01 | 169280 |
| f-LIF | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| f-LIF+mean | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| pop LIF all | 0.2910 | 0.4134 | 0.1769 | 0.4033 | 0.3212 | 0.36151 | 28.50 | 170944 |
| pop LIF all+mean | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| pop LIF mixer | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| pop LIF mixer+mean | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| pop f-LIF all | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| pop f-LIF all+mean | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| pop f-LIF mixer | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| pop f-LIF mixer+mean | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| SF-de | 0.2903 | 0.4223 | 0.1767 | 0.4044 | 0.3234 | 0.36087 | 10.48 | 169656 |
| SF-sj | 0.2903 | 0.4223 | 0.1767 | 0.4044 | 0.3234 | 0.36087 | 10.48 | 169656 |

## Comparisons

| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % | validation L1 R % |
|---|--:|--:|---|--:|---|--:|--:|
| B1 pop LIF mixer vs LIF | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| B2 pop LIF all vs LIF | -0.94 | 10/12 | no clear difference | 2.78 | -1.41, -1.98, -0.34, -0.02 | -0.71 | +0.01 |
| B3 pop LIF mixer vs pop LIF all | +0.96 | 2/12 | no clear difference | 0.36 | +1.45, +2.03, +0.34, +0.03 | +0.72 | -0.01 |
| B4 pop f-LIF mixer vs f-LIF | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| B5 pop f-LIF all vs f-LIF | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| B6 pop f-LIF mixer vs pop f-LIF all | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| C1 f-LIF+mean vs f-LIF | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| C2 pop LIF all+mean vs pop LIF all | +0.96 | 2/12 | no clear difference | 0.36 | +1.45, +2.03, +0.34, +0.03 | +0.72 | -0.01 |
| C3 pop LIF mixer+mean vs pop LIF mixer | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| C4 pop f-LIF all+mean vs pop f-LIF all | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| C5 pop f-LIF mixer+mean vs pop f-LIF mixer | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| (report) f-LIF vs LIF | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| (report) LIF+mean vs LIF (= 11-18 V2) | -0.65 | 9/12 | no clear difference | 1.27 | -1.99, -0.45, -0.12, -0.03 | -0.35 | +0.27 |
| (report, mean) pop LIF mixer vs LIF | +0.67 | 3/12 | no clear difference | 0.79 | +2.04, +0.47, +0.12, +0.03 | +0.36 | -0.26 |
| (report, mean) pop LIF all vs LIF | +0.67 | 3/12 | no clear difference | 0.79 | +2.04, +0.47, +0.12, +0.03 | +0.36 | -0.26 |
| (report, mean) pop LIF mixer vs pop LIF all | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| (report, mean) pop f-LIF mixer vs f-LIF | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| (report, mean) pop f-LIF all vs f-LIF | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| (report, mean) pop f-LIF mixer vs pop f-LIF all | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| (report) LIF vs SF-de | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) LIF+mean vs SF-de | -0.21 | 5/12 | no clear difference | 1.24 | -0.35, -0.56, +0.34, -0.26 | +0.09 | +0.43 |
| (report) f-LIF vs SF-de | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) f-LIF+mean vs SF-de | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) pop LIF all vs SF-de | -0.50 | 8/12 | no clear difference | 2.73 | +0.23, -2.11, +0.13, -0.26 | -0.27 | +0.18 |
| (report) pop LIF all+mean vs SF-de | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) pop LIF mixer vs SF-de | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) pop LIF mixer+mean vs SF-de | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) pop f-LIF all vs SF-de | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) pop f-LIF all+mean vs SF-de | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) pop f-LIF mixer vs SF-de | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) pop f-LIF mixer+mean vs SF-de | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) LIF vs SF-sj | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) pop LIF mixer vs SF-sj | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) pop f-LIF mixer vs SF-sj | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) SF-de vs SF-sj | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |

Pairs r % (test MSE), seed 42, 7, 2026 per cell (verdict comparisons):

- B1 pop LIF mixer vs LIF: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00
- B2 pop LIF all vs LIF: ETTh2 H96 -3.23, +0.12, -1.11; ETTh2 H720 -3.41, -1.25, -1.28; ETTm2 H96 -0.31, -0.01, -0.70; ETTm2 H720 +0.36, -0.15, -0.29
- B3 pop LIF mixer vs pop LIF all: ETTh2 H96 +3.34, -0.12, +1.12; ETTh2 H720 +3.53, +1.26, +1.30; ETTm2 H96 +0.31, +0.01, +0.70; ETTm2 H720 -0.36, +0.15, +0.29
- B4 pop f-LIF mixer vs f-LIF: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00
- B5 pop f-LIF all vs f-LIF: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00
- B6 pop f-LIF mixer vs pop f-LIF all: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00
- C1 f-LIF+mean vs f-LIF: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00
- C2 pop LIF all+mean vs pop LIF all: ETTh2 H96 +3.34, -0.12, +1.12; ETTh2 H720 +3.53, +1.26, +1.30; ETTm2 H96 +0.31, +0.01, +0.70; ETTm2 H720 -0.36, +0.15, +0.29
- C3 pop LIF mixer+mean vs pop LIF mixer: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00
- C4 pop f-LIF all+mean vs pop f-LIF all: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00
- C5 pop f-LIF mixer+mean vs pop f-LIF mixer: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00

## Report only: stopping and firing rates (seed mean)

| cell | arm | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | energy µJ | time s (seeds) |
|---|---|---|---|--:|--:|--:|---|
| ETTh2 H96 | LIF | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 | 461, 404, 498 |
| ETTh2 H96 | LIF+mean | 0, 2, 0 | 11, 13, 11 | 0.2952 | 0.0709 | 11.78 | 294, 412, 309 |
| ETTh2 H96 | f-LIF | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 | 461, 404, 498 |
| ETTh2 H96 | f-LIF+mean | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 | 461, 404, 498 |
| ETTh2 H96 | pop LIF all | 0, 1, 1 | 11, 12, 12 | 0.2892 | 0.0715 | 26.63 | 598, 790, 806 |
| ETTh2 H96 | pop LIF all+mean | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 | 461, 404, 498 |
| ETTh2 H96 | pop LIF mixer | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 | 461, 404, 498 |
| ETTh2 H96 | pop LIF mixer+mean | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 | 461, 404, 498 |
| ETTh2 H96 | pop f-LIF all | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 | 461, 404, 498 |
| ETTh2 H96 | pop f-LIF all+mean | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 | 461, 404, 498 |
| ETTh2 H96 | pop f-LIF mixer | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 | 461, 404, 498 |
| ETTh2 H96 | pop f-LIF mixer+mean | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 | 461, 404, 498 |
| ETTh2 H96 | SF-de | 3, 1, 7 | 14, 12, 18 | 0.2879 | 0.0505 | 9.30 | 263, 235, 234 |
| ETTh2 H96 | SF-sj | 3, 1, 7 | 14, 12, 18 | 0.2879 | 0.0505 | 9.30 | 263, 235, 234 |
| ETTh2 H720 | LIF | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 | 377, 342, 416 |
| ETTh2 H720 | LIF+mean | 3, 0, 2 | 14, 11, 13 | 0.4228 | 0.0716 | 13.52 | 343, 433, 339 |
| ETTh2 H720 | f-LIF | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 | 377, 342, 416 |
| ETTh2 H720 | f-LIF+mean | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 | 377, 342, 416 |
| ETTh2 H720 | pop LIF all | 1, 0, 2 | 12, 11, 13 | 0.4201 | 0.0678 | 28.32 | 696, 697, 814 |
| ETTh2 H720 | pop LIF all+mean | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 | 377, 342, 416 |
| ETTh2 H720 | pop LIF mixer | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 | 377, 342, 416 |
| ETTh2 H720 | pop LIF mixer+mean | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 | 377, 342, 416 |
| ETTh2 H720 | pop f-LIF all | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 | 377, 342, 416 |
| ETTh2 H720 | pop f-LIF all+mean | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 | 377, 342, 416 |
| ETTh2 H720 | pop f-LIF mixer | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 | 377, 342, 416 |
| ETTh2 H720 | pop f-LIF mixer+mean | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 | 377, 342, 416 |
| ETTh2 H720 | SF-de | 3, 3, 4 | 14, 14, 15 | 0.4223 | 0.0493 | 10.79 | 360, 363, 359 |
| ETTh2 H720 | SF-sj | 3, 3, 4 | 14, 14, 15 | 0.4223 | 0.0493 | 10.79 | 360, 363, 359 |
| ETTm2 H96 | LIF | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 | 3829, 2399, 2162 |
| ETTm2 H96 | LIF+mean | 15, 11, 21 | 26, 22, 32 | 0.1772 | 0.0843 | 12.51 | 3745, 3582, 2872 |
| ETTm2 H96 | f-LIF | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 | 3829, 2399, 2162 |
| ETTm2 H96 | f-LIF+mean | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 | 3829, 2399, 2162 |
| ETTm2 H96 | pop LIF all | 8, 5, 6 | 19, 16, 17 | 0.1768 | 0.0799 | 28.62 | 3694, 3753, 3355 |
| ETTm2 H96 | pop LIF all+mean | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 | 3829, 2399, 2162 |
| ETTm2 H96 | pop LIF mixer | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 | 3829, 2399, 2162 |
| ETTm2 H96 | pop LIF mixer+mean | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 | 3829, 2399, 2162 |
| ETTm2 H96 | pop f-LIF all | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 | 3829, 2399, 2162 |
| ETTm2 H96 | pop f-LIF all+mean | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 | 3829, 2399, 2162 |
| ETTm2 H96 | pop f-LIF mixer | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 | 3829, 2399, 2162 |
| ETTm2 H96 | pop f-LIF mixer+mean | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 | 3829, 2399, 2162 |
| ETTm2 H96 | SF-de | 11, 11, 4 | 22, 22, 15 | 0.1770 | 0.0695 | 10.29 | 606, 578, 692 |
| ETTm2 H96 | SF-sj | 11, 11, 4 | 22, 22, 15 | 0.1770 | 0.0695 | 10.29 | 606, 578, 692 |
| ETTm2 H720 | LIF | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 | 1729, 1905, 1600 |
| ETTm2 H720 | LIF+mean | 2, 3, 2 | 13, 14, 13 | 0.4079 | 0.0729 | 14.23 | 1888, 2065, 1859 |
| ETTm2 H720 | f-LIF | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 | 1729, 1905, 1600 |
| ETTm2 H720 | f-LIF+mean | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 | 1729, 1905, 1600 |
| ETTm2 H720 | pop LIF all | 2, 1, 1 | 13, 12, 12 | 0.4104 | 0.0705 | 30.43 | 2649, 2528, 2341 |
| ETTm2 H720 | pop LIF all+mean | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 | 1729, 1905, 1600 |
| ETTm2 H720 | pop LIF mixer | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 | 1729, 1905, 1600 |
| ETTm2 H720 | pop LIF mixer+mean | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 | 1729, 1905, 1600 |
| ETTm2 H720 | pop f-LIF all | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 | 1729, 1905, 1600 |
| ETTm2 H720 | pop f-LIF all+mean | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 | 1729, 1905, 1600 |
| ETTm2 H720 | pop f-LIF mixer | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 | 1729, 1905, 1600 |
| ETTm2 H720 | pop f-LIF mixer+mean | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 | 1729, 1905, 1600 |
| ETTm2 H720 | SF-de | 2, 1, 1 | 13, 12, 12 | 0.4139 | 0.0539 | 11.54 | 428, 437, 344 |
| ETTm2 H720 | SF-sj | 2, 1, 1 | 13, 12, 12 | 0.4139 | 0.0539 | 11.54 | 428, 437, 344 |
# Part 4, ETT input 336: SF-de and pop LIF mixer-only

Rules and sources: module docstring of summarize_pn.py. Seeds 42 / 7 / 2026.

## Test MSE (seed mean) and energy

| arm | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean | mean best validation L1 | energy µJ (4-cell mean) | params (ETTh2 H96) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| LIF 336 | 0.2964 | 0.3916 | 0.1726 | 0.3778 | 0.3096 | 0.36450 | 34.69 | 537920 |
| pop LIF all 336 | 0.2976 | 0.4202 | 0.1696 | 0.3631 | 0.3126 | 0.36252 | 98.68 | 539584 |
| SF-sj 336 | 0.3094 | 0.3983 | 0.1715 | 0.3673 | 0.3116 | 0.36412 | 35.13 | 539256 |
| pop LIF mixer 336 | 0.2964 | 0.3916 | 0.1726 | 0.3778 | 0.3096 | 0.36450 | 34.69 | 537920 |
| SF-de 336 | 0.3094 | 0.3983 | 0.1715 | 0.3673 | 0.3116 | 0.36412 | 35.13 | 539256 |

## Comparisons

| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % | validation L1 R % |
|---|--:|--:|---|--:|---|--:|--:|
| D1 LIF 336 vs SF-de 336 | -0.57 | 6/12 | no clear difference | 0.99 | -4.08, -1.69, +0.62, +2.86 | -0.34 | +0.05 |
| D2 pop LIF mixer 336 vs LIF 336 | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| D3 pop LIF mixer 336 vs pop LIF all 336 | -0.35 | 5/12 | no clear difference | 0.35 | -0.40, -6.79, +1.76, +4.03 | -0.36 | +0.53 |
| (report) pop LIF mixer 336 vs SF-de 336 | -0.57 | 6/12 | no clear difference | 0.99 | -4.08, -1.69, +0.62, +2.86 | -0.34 | +0.05 |
| (report) SF-de 336 vs SF-sj 336 | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| (report) LIF 336 vs SF-sj 336 (= 11-18 P2) | -0.57 | 6/12 | no clear difference | 0.99 | -4.08, -1.69, +0.62, +2.86 | -0.34 | +0.05 |

Pairs r % (test MSE), seed 42, 7, 2026 per cell (verdict comparisons):

- D1 LIF 336 vs SF-de 336: ETTh2 H96 +1.84, -8.92, -5.16; ETTh2 H720 -1.00, -2.37, -1.69; ETTm2 H96 +1.38, -0.44, +0.92; ETTm2 H720 +3.44, +2.31, +2.82
- D2 pop LIF mixer 336 vs LIF 336: ETTh2 H96 +0.00, +0.00, +0.00; ETTh2 H720 +0.00, +0.00, +0.00; ETTm2 H96 +0.00, +0.00, +0.00; ETTm2 H720 +0.00, +0.00, +0.00
- D3 pop LIF mixer 336 vs pop LIF all 336: ETTh2 H96 -1.02, -1.21, +1.02; ETTh2 H720 -7.70, -4.97, -7.70; ETTm2 H96 +0.24, +2.16, +2.90; ETTm2 H720 +5.57, +3.47, +3.05

## Report only: stopping and firing rates (seed mean)

| cell | arm | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | energy µJ | time s (seeds) |
|---|---|---|---|--:|--:|--:|---|
| ETTh2 H96 | LIF 336 | 1, 1, 1 | 12, 12, 12 | 0.3092 | 0.0736 | 31.38 | 1481, 1520, 1453 |
| ETTh2 H96 | pop LIF all 336 | 2, 1, 0 | 13, 12, 11 | 0.3239 | 0.0779 | 92.94 | 3772, 4011, 3556 |
| ETTh2 H96 | SF-sj 336 | 1, 4, 2 | 12, 15, 13 | 0.3150 | 0.0549 | 32.26 | 198, 241, 216 |
| ETTh2 H96 | pop LIF mixer 336 | 1, 1, 1 | 12, 12, 12 | 0.3092 | 0.0736 | 31.38 | 1481, 1520, 1453 |
| ETTh2 H96 | SF-de 336 | 1, 4, 2 | 12, 15, 13 | 0.3150 | 0.0549 | 32.26 | 198, 241, 216 |
| ETTh2 H720 | LIF 336 | 0, 0, 0 | 11, 11, 11 | 0.4019 | 0.0732 | 34.78 | 1336, 1090, 1154 |
| ETTh2 H720 | pop LIF all 336 | 3, 7, 3 | 14, 18, 14 | 0.4277 | 0.0843 | 96.98 | 4746, 4723, 4785 |
| ETTh2 H720 | SF-sj 336 | 0, 0, 0 | 11, 11, 11 | 0.4113 | 0.0461 | 35.02 | 185, 161, 189 |
| ETTh2 H720 | pop LIF mixer 336 | 0, 0, 0 | 11, 11, 11 | 0.4019 | 0.0732 | 34.78 | 1336, 1090, 1154 |
| ETTh2 H720 | SF-de 336 | 0, 0, 0 | 11, 11, 11 | 0.4113 | 0.0461 | 35.02 | 185, 161, 189 |
| ETTm2 H96 | LIF 336 | 7, 13, 12 | 18, 24, 23 | 0.1731 | 0.0804 | 34.37 | 8346, 11360, 11419 |
| ETTm2 H96 | pop LIF all 336 | 3, 3, 2 | 14, 14, 13 | 0.1737 | 0.0693 | 99.74 | 26194, 20408, 18344 |
| ETTm2 H96 | SF-sj 336 | 2, 3, 2 | 13, 14, 13 | 0.1758 | 0.0588 | 35.30 | 942, 948, 947 |
| ETTm2 H96 | pop LIF mixer 336 | 7, 13, 12 | 18, 24, 23 | 0.1731 | 0.0804 | 34.37 | 8346, 11360, 11419 |
| ETTm2 H96 | SF-de 336 | 2, 3, 2 | 13, 14, 13 | 0.1758 | 0.0588 | 35.30 | 942, 948, 947 |
| ETTm2 H720 | LIF 336 | 1, 2, 0 | 12, 13, 11 | 0.3808 | 0.0698 | 38.21 | 5140, 6447, 5528 |
| ETTm2 H720 | pop LIF all 336 | 0, 0, 0 | 11, 11, 11 | 0.3851 | 0.0646 | 105.05 | 17819, 17147, 22989 |
| ETTm2 H720 | SF-sj 336 | 0, 0, 0 | 11, 11, 11 | 0.3790 | 0.0460 | 37.94 | 943, 782, 813 |
| ETTm2 H720 | pop LIF mixer 336 | 1, 2, 0 | 12, 13, 11 | 0.3808 | 0.0698 | 38.21 | 5140, 6447, 5528 |
| ETTm2 H720 | SF-de 336 | 0, 0, 0 | 11, 11, 11 | 0.3790 | 0.0460 | 37.94 | 943, 782, 813 |
# Parts 2 and 3, MIT-BIH: SpkMixer neuron x placement x mean pooling

Rules and sources: module docstring of summarize_pn.py. Seeds 42 / 7 / 2026; last-epoch weights on the 30 % test file.

## Test values (seed mean)

| arm | macro-F1 % | per seed 42, 7, 2026 | accuracy % | F1 N / S / V / F / Q | rho_hippo | energy µJ | params | time s (seeds) |
|---|--:|---|--:|---|--:|--:|--:|---|
| LIF | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 | 1311, 1297, 1311 |
| LIF+mean | 91.73 | 91.42, 92.19, 91.58 | 98.53 | 99.22 / 85.45 / 96.27 / 78.76 / 98.96 | 0.0850 | 20.56 | 1200768 | 1190, 834, 1012 |
| f-LIF | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 | 1311, 1297, 1311 |
| f-LIF+mean | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 | 1311, 1297, 1311 |
| pop LIF all | 91.40 | 91.50, 91.73, 90.96 | 98.49 | 99.20 / 85.24 / 96.28 / 77.39 / 98.88 | 0.1027 | 19.98 | 1214592 | 5979, 5971, 5971 |
| pop LIF all+mean | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 | 1311, 1297, 1311 |
| pop LIF mixer | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 | 1311, 1297, 1311 |
| pop LIF mixer+mean | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 | 1311, 1297, 1311 |
| pop f-LIF all | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 | 1311, 1297, 1311 |
| pop f-LIF all+mean | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 | 1311, 1297, 1311 |
| pop f-LIF mixer | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 | 1311, 1297, 1311 |
| pop f-LIF mixer+mean | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 | 1311, 1297, 1311 |
| SF-de | 91.78 | 92.10, 91.74, 91.50 | 98.48 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 0.1275 | 9.61 | 904488 | 420, 408, 410 |
| SF-sj | 91.78 | 92.10, 91.74, 91.50 | 98.48 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 0.1275 | 9.61 | 904488 | 420, 408, 410 |

## Comparisons

| comparison | mean %p | seeds higher | per seed 42, 7, 2026 | verdict | energy A/B |
|---|--:|--:|---|---|--:|
| B1 pop LIF mixer vs LIF | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| B2 pop LIF all vs LIF | -0.59 | 0/3 | -0.38, -0.39, -0.98 | worse | 2.12 |
| B3 pop LIF mixer vs pop LIF all | +0.59 | 3/3 | +0.38, +0.39, +0.98 | better | 0.47 |
| B4 pop f-LIF mixer vs f-LIF | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| B5 pop f-LIF all vs f-LIF | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| B6 pop f-LIF mixer vs pop f-LIF all | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| C1 f-LIF+mean vs f-LIF | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| C2 pop LIF all+mean vs pop LIF all | +0.59 | 3/3 | +0.38, +0.39, +0.98 | better | 0.47 |
| C3 pop LIF mixer+mean vs pop LIF mixer | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| C4 pop f-LIF all+mean vs pop f-LIF all | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| C5 pop f-LIF mixer+mean vs pop f-LIF mixer | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| (report) f-LIF vs LIF | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| (report) LIF+mean vs LIF (= 11-18 V2) | -0.26 | 1/3 | -0.46, +0.06, -0.36 | no clear difference | 2.18 |
| (report, mean) pop LIF mixer vs LIF | +0.26 | 2/3 | +0.46, -0.06, +0.36 | no clear difference | 0.46 |
| (report, mean) pop LIF all vs LIF | +0.26 | 2/3 | +0.46, -0.06, +0.36 | no clear difference | 0.46 |
| (report, mean) pop LIF mixer vs pop LIF all | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| (report, mean) pop f-LIF mixer vs f-LIF | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| (report, mean) pop f-LIF all vs f-LIF | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| (report, mean) pop f-LIF mixer vs pop f-LIF all | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| (report) LIF vs SF-de | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) LIF+mean vs SF-de | -0.05 | 2/3 | -0.68, +0.45, +0.08 | no clear difference | 2.14 |
| (report) f-LIF vs SF-de | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) f-LIF+mean vs SF-de | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) pop LIF all vs SF-de | -0.38 | 0/3 | -0.60, -0.01, -0.54 | no clear difference | 2.08 |
| (report) pop LIF all+mean vs SF-de | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) pop LIF mixer vs SF-de | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) pop LIF mixer+mean vs SF-de | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) pop f-LIF all vs SF-de | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) pop f-LIF all+mean vs SF-de | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) pop f-LIF mixer vs SF-de | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) pop f-LIF mixer+mean vs SF-de | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) LIF vs SF-sj | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) pop LIF mixer vs SF-sj | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) pop f-LIF mixer vs SF-sj | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) SF-de vs SF-sj | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
# SF-de learning-rate selection (validation only)

- ETTh2: selected 0.0005; score (mean over H96 / H720 of the seed-mean best validation L1 relative to lr 5e-4): 0.0001 1.00286, 0.00025 1.00226, 0.0005 1.00000, 0.001 1.00238, 0.002 1.00564
- ETTm2: selected 0.00025; score (mean over H96 / H720 of the seed-mean best validation L1 relative to lr 5e-4): 0.0001 0.99761, 0.00025 0.99753, 0.0005 1.00000, 0.001 0.99971, 0.002 1.00432
- MIT-BIH: selected 0.005; seed-mean last-epoch validation macro-F1 on fold 0: 0.0025 91.929, 0.005 92.597, 0.01 92.432, 0.02 92.286

../../../../../../../tmp/claude-1001/-home-yschoi-CLS-spiking-transformer-Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning-NSMT/545f439f-5e1a-4ae0-89ef-16217459402b/scratchpad/pn/dry/flif-v1-ett-sfde/selected_lr.json: {'ETTh2': '0.0005', 'ETTm2': '0.00025'}
../../../../../../../tmp/claude-1001/-home-yschoi-CLS-spiking-transformer-Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning-NSMT/545f439f-5e1a-4ae0-89ef-16217459402b/scratchpad/pn/dry/flif-v1-mitbih-sfde/selected_lr.json: {'MITBIH': '0.005'}
# Part 1, ETT: LIF SpkMixer vs the Spikformer-style model with the same spikeDE LIF (SF-de)

Rules and sources: module docstring of summarize_pn.py. Seeds 42 / 7 / 2026.

## Test MSE (seed mean) and energy

| arm | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean | mean best validation L1 | energy µJ (4-cell mean) | params (ETTh2 H96) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| LIF SpkMixer | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| SF-de | 0.2903 | 0.4223 | 0.1767 | 0.4044 | 0.3234 | 0.36087 | 10.48 | 169656 |
| SF-sj | 0.2903 | 0.4223 | 0.1767 | 0.4044 | 0.3234 | 0.36087 | 10.48 | 169656 |

## Comparisons

| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % | validation L1 R % |
|---|--:|--:|---|--:|---|--:|--:|
| A1 LIF SpkMixer vs SF-de | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) SF-de vs SF-sj | +0.00 | 0/12 | no clear difference | 1.00 | +0.00, +0.00, +0.00, +0.00 | +0.00 | +0.00 |
| (report) LIF SpkMixer vs SF-sj | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |

Pairs r % (test MSE), seed 42, 7, 2026 per cell (verdict comparisons):

- A1 LIF SpkMixer vs SF-de: ETTh2 H96 +2.37, +0.39, +2.25; ETTh2 H720 +0.83, -0.55, -0.64; ETTm2 H96 +0.19, +0.43, +0.77; ETTm2 H720 -0.89, +0.07, +0.12

## Report only: stopping and firing rates (seed mean)

| cell | arm | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | energy µJ | time s (seeds) |
|---|---|---|---|--:|--:|--:|---|
| ETTh2 H96 | LIF SpkMixer | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 | 461, 404, 498 |
| ETTh2 H96 | SF-de | 3, 1, 7 | 14, 12, 18 | 0.2879 | 0.0505 | 9.30 | 263, 235, 234 |
| ETTh2 H96 | SF-sj | 3, 1, 7 | 14, 12, 18 | 0.2879 | 0.0505 | 9.30 | 263, 235, 234 |
| ETTh2 H720 | LIF SpkMixer | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 | 377, 342, 416 |
| ETTh2 H720 | SF-de | 3, 3, 4 | 14, 14, 15 | 0.4223 | 0.0493 | 10.79 | 360, 363, 359 |
| ETTh2 H720 | SF-sj | 3, 3, 4 | 14, 14, 15 | 0.4223 | 0.0493 | 10.79 | 360, 363, 359 |
| ETTm2 H96 | LIF SpkMixer | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 | 3829, 2399, 2162 |
| ETTm2 H96 | SF-de | 11, 11, 4 | 22, 22, 15 | 0.1770 | 0.0695 | 10.29 | 606, 578, 692 |
| ETTm2 H96 | SF-sj | 11, 11, 4 | 22, 22, 15 | 0.1770 | 0.0695 | 10.29 | 606, 578, 692 |
| ETTm2 H720 | LIF SpkMixer | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 | 1729, 1905, 1600 |
| ETTm2 H720 | SF-de | 2, 1, 1 | 13, 12, 12 | 0.4139 | 0.0539 | 11.54 | 428, 437, 344 |
| ETTm2 H720 | SF-sj | 2, 1, 1 | 13, 12, 12 | 0.4139 | 0.0539 | 11.54 | 428, 437, 344 |
# Part 1, MIT-BIH: LIF SpkMixer vs the Spikformer-style model with the same spikeDE LIF (SF-de)

Rules and sources: module docstring of summarize_pn.py. Seeds 42 / 7 / 2026; last-epoch weights on the 30 % test file.

## Test values (seed mean)

| arm | macro-F1 % | per seed 42, 7, 2026 | accuracy % | F1 N / S / V / F / Q | rho_hippo | energy µJ | params | time s (seeds) |
|---|--:|---|--:|---|--:|--:|--:|---|
| LIF SpkMixer | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 | 1311, 1297, 1311 |
| SF-de | 91.78 | 92.10, 91.74, 91.50 | 98.48 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 0.1275 | 9.61 | 904488 | 420, 408, 410 |
| SF-sj | 91.78 | 92.10, 91.74, 91.50 | 98.48 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 0.1275 | 9.61 | 904488 | 420, 408, 410 |

## Comparisons

| comparison | mean %p | seeds higher | per seed 42, 7, 2026 | verdict | energy A/B |
|---|--:|--:|---|---|--:|
| A1 LIF SpkMixer vs SF-de | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) SF-de vs SF-sj | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| (report) LIF SpkMixer vs SF-sj | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
