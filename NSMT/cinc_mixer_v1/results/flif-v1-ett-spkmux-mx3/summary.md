# Mixer variants of the LIF SpkMixer, ETT

Rules and sources: module docstring of summarize_mx3.py. Seeds 42 / 7 / 2026. Tuned LIF SpkMixer settings, not re-tuned.

## Test MSE (seed mean) and energy

| arm | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean | mean best validation L1 | energy µJ (4-cell mean) | params (ETTh2 H96) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| base | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 10.28 | 169280 |
| max | 0.2922 | 0.4141 | 0.1766 | 0.4037 | 0.3217 | 0.36188 | 6.87 | 169280 |
| mean | 0.2893 | 0.4199 | 0.1773 | 0.4033 | 0.3225 | 0.36242 | 13.01 | 169280 |
| L2 | 0.2913 | 0.4204 | 0.1782 | 0.4041 | 0.3235 | 0.36182 | 12.40 | 177728 |
| MS | 0.2912 | 0.4209 | 0.1778 | 0.4043 | 0.3235 | 0.36172 | 12.54 | 177728 |
| subtract | 0.2983 | 0.4174 | 0.1774 | 0.4040 | 0.3243 | 0.36166 | 10.24 | 169280 |
| zero | 0.2949 | 0.4200 | 0.1773 | 0.4041 | 0.3241 | 0.36133 | 10.21 | 169280 |
| Spikformer-style | 0.2903 | 0.4223 | 0.1767 | 0.4044 | 0.3234 | 0.36087 | 10.48 | 169656 |

## Pre-registered comparisons

| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % | validation L1 R % |
|---|--:|--:|---|--:|---|--:|--:|
| V1 max vs base | -0.81 | 8/12 | no clear difference | 0.67 | -0.99, -1.82, -0.50, +0.06 | -0.34 | +0.09 |
| V2 mean vs base | -0.65 | 9/12 | no clear difference | 1.27 | -1.99, -0.45, -0.12, -0.03 | -0.35 | +0.27 |
| V3 MS vs base | -0.30 | 4/12 | no clear difference | 1.22 | -1.36, -0.22, +0.16, +0.21 | -0.18 | +0.06 |
| V4 MS vs L2 | -0.01 | 6/12 | no clear difference | 1.01 | -0.03, +0.13, -0.21, +0.06 | -0.03 | -0.03 |
| V5 subtract vs base | +0.01 | 4/12 | no clear difference | 1.00 | +1.06, -1.04, -0.10, +0.13 | +0.01 | +0.03 |
| V6 zero vs base | -0.13 | 5/12 | no clear difference | 0.99 | -0.10, -0.43, -0.14, +0.16 | -0.02 | -0.04 |
| (report) max vs mean | -0.16 | 5/12 | no clear difference | 0.53 | +1.02, -1.37, -0.38, +0.09 | +0.02 | -0.18 |
| (report) L2 vs base | -0.29 | 4/12 | no clear difference | 1.21 | -1.33, -0.34, +0.37, +0.16 | -0.15 | +0.09 |
| (report) subtract vs zero | +0.14 | 5/12 | no clear difference | 1.00 | +1.17, -0.61, +0.04, -0.03 | +0.04 | +0.07 |
| (report) base vs Spikformer-style | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 | +0.17 |
| (report) max vs Spikformer-style | -0.37 | 6/12 | no clear difference | 0.65 | +0.66, -1.94, -0.04, -0.17 | +0.10 | +0.25 |
| (report) mean vs Spikformer-style | -0.21 | 5/12 | no clear difference | 1.24 | -0.35, -0.56, +0.34, -0.26 | +0.09 | +0.43 |
| (report) L2 vs Spikformer-style | +0.15 | 4/12 | no clear difference | 1.18 | +0.31, -0.46, +0.84, -0.08 | +0.29 | +0.26 |
| (report) MS vs Spikformer-style | +0.14 | 5/12 | no clear difference | 1.20 | +0.28, -0.34, +0.63, -0.02 | +0.26 | +0.23 |
| (report) subtract vs Spikformer-style | +0.46 | 4/12 | no clear difference | 0.98 | +2.75, -1.16, +0.37, -0.10 | +0.46 | +0.20 |
| (report) zero vs Spikformer-style | +0.32 | 4/12 | no clear difference | 0.97 | +1.57, -0.56, +0.33, -0.07 | +0.42 | +0.13 |

Pairs r % (test MSE), seed 42, 7, 2026 per cell:

- V1 max vs base: ETTh2 H96 -2.21, -0.04, -0.73; ETTh2 H720 -1.09, -2.04, -2.35; ETTm2 H96 +0.06, -0.44, -1.12; ETTm2 H720 +0.05, +0.07, +0.06
- V2 mean vs base: ETTh2 H96 -3.50, -0.74, -1.72; ETTh2 H720 +1.07, -0.26, -2.16; ETTm2 H96 +0.02, -0.10, -0.29; ETTm2 H720 +0.01, -0.02, -0.09
- V3 MS vs base: ETTh2 H96 -1.58, +0.30, -2.80; ETTh2 H720 -1.05, +0.23, +0.17; ETTm2 H96 +0.45, +0.02, +0.00; ETTm2 H720 +0.26, +0.41, -0.03
- V4 MS vs L2: ETTh2 H96 -0.23, -0.10, +0.23; ETTh2 H720 -1.50, +0.73, +1.15; ETTm2 H96 +0.22, -0.76, -0.08; ETTm2 H720 +0.14, +0.26, -0.23
- V5 subtract vs base: ETTh2 H96 +1.84, +1.12, +0.22; ETTh2 H720 -1.39, +0.02, -1.75; ETTm2 H96 +0.54, -0.28, -0.56; ETTm2 H720 +0.19, +0.07, +0.13
- V6 zero vs base: ETTh2 H96 +1.37, -0.02, -1.66; ETTh2 H720 -0.97, +0.53, -0.86; ETTm2 H96 +0.14, +0.05, -0.60; ETTm2 H720 +0.17, +0.22, +0.09

## Report only: stopping and firing rates (seed mean)

| cell | arm | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | energy µJ |
|---|---|---|---|--:|--:|--:|
| ETTh2 H96 | base | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 |
| ETTh2 H96 | max | 1, 1, 3 | 12, 12, 14 | 0.2973 | 0.0749 | 5.66 |
| ETTh2 H96 | mean | 0, 2, 0 | 11, 13, 11 | 0.2952 | 0.0709 | 11.78 |
| ETTh2 H96 | L2 | 1, 1, 0 | 12, 12, 11 | 0.2950 | 0.0692 | 11.17 |
| ETTh2 H96 | MS | 1, 2, 0 | 12, 13, 11 | 0.2975 | 0.0663 | 11.27 |
| ETTh2 H96 | subtract | 3, 1, 7 | 14, 12, 18 | 0.2949 | 0.0697 | 9.02 |
| ETTh2 H96 | zero | 3, 1, 1 | 14, 12, 12 | 0.2952 | 0.0698 | 9.01 |
| ETTh2 H720 | base | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 |
| ETTh2 H720 | max | 1, 3, 2 | 12, 14, 13 | 0.4206 | 0.0686 | 7.33 |
| ETTh2 H720 | mean | 3, 0, 2 | 14, 11, 13 | 0.4228 | 0.0716 | 13.52 |
| ETTh2 H720 | L2 | 1, 0, 2 | 12, 11, 13 | 0.4163 | 0.0621 | 12.67 |
| ETTh2 H720 | MS | 1, 0, 2 | 12, 11, 13 | 0.4183 | 0.0601 | 12.82 |
| ETTh2 H720 | subtract | 1, 0, 2 | 12, 11, 13 | 0.4182 | 0.0624 | 10.65 |
| ETTh2 H720 | zero | 1, 0, 2 | 12, 11, 13 | 0.4185 | 0.0637 | 10.66 |
| ETTm2 H96 | base | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 |
| ETTm2 H96 | max | 9, 14, 12 | 20, 25, 23 | 0.1766 | 0.0839 | 6.41 |
| ETTm2 H96 | mean | 15, 11, 21 | 26, 22, 32 | 0.1772 | 0.0843 | 12.51 |
| ETTm2 H96 | L2 | 6, 6, 6 | 17, 17, 17 | 0.1779 | 0.0802 | 12.08 |
| ETTm2 H96 | MS | 6, 14, 13 | 17, 25, 24 | 0.1776 | 0.0823 | 12.25 |
| ETTm2 H96 | subtract | 6, 11, 15 | 17, 22, 26 | 0.1771 | 0.0810 | 9.81 |
| ETTm2 H96 | zero | 11, 11, 15 | 22, 22, 26 | 0.1773 | 0.0813 | 9.79 |
| ETTm2 H720 | base | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 |
| ETTm2 H720 | max | 2, 4, 3 | 13, 15, 14 | 0.4095 | 0.0744 | 8.08 |
| ETTm2 H720 | mean | 2, 3, 2 | 13, 14, 13 | 0.4079 | 0.0729 | 14.23 |
| ETTm2 H720 | L2 | 2, 1, 3 | 13, 12, 14 | 0.4101 | 0.0719 | 13.67 |
| ETTm2 H720 | MS | 2, 3, 2 | 13, 14, 13 | 0.4091 | 0.0753 | 13.83 |
| ETTm2 H720 | subtract | 2, 2, 3 | 13, 13, 14 | 0.4087 | 0.0729 | 11.48 |
| ETTm2 H720 | zero | 2, 4, 3 | 13, 15, 14 | 0.4099 | 0.0733 | 11.38 |
