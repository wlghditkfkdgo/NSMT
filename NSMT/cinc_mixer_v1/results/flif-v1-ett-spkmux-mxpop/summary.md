# Mixer-only population, ETT input 96 (user decisions 11-19): population neuron only in the directional stacks + fusion

Rules and sources: module docstring of summarize_pn.py. Seeds 42 / 7 / 2026. Energy columns: the syops rule as published (user 11-19; population neuron input ACs added from the recorder).

## Test MSE (seed mean) and energy

| arm | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean | mean best validation L1 | energy µJ (4-cell mean) | params (ETTh2 H96) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| LIF | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.36145 | 29.30 | 169280 |
| LIF+mean | 0.2893 | 0.4199 | 0.1773 | 0.4033 | 0.3225 | 0.36242 | 32.02 | 169280 |
| pop LIF all | 0.2910 | 0.4134 | 0.1769 | 0.4033 | 0.3212 | 0.36151 | 47.46 | 170944 |
| pop LIF mixer | 0.2889 | 0.4198 | 0.1776 | 0.4044 | 0.3226 | 0.36260 | 40.12 | 170304 |
| pop LIF mixer+mean | 0.2890 | 0.4220 | 0.1777 | 0.4049 | 0.3234 | 0.36205 | 34.71 | 170304 |
| pop f-LIF mixer | 0.2889 | 0.4196 | 0.1783 | 0.4053 | 0.3230 | 0.36308 | 58.01 | 170304 |
| pop f-LIF mixer+mean | 0.2929 | 0.4188 | 0.1785 | 0.4061 | 0.3241 | 0.36303 | 39.25 | 170304 |
| SF-de | 0.2921 | 0.4229 | 0.1768 | 0.4036 | 0.3238 | 0.36126 | 29.68 | 169656 |

## Comparisons

| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % | validation L1 R % |
|---|--:|--:|---|--:|---|--:|--:|
| B1 pop LIF mixer vs LIF | -0.59 | 6/12 | no clear difference | 1.51 | -2.13, -0.48, +0.02, +0.23 | -0.28 | +0.36 |
| B3 pop LIF mixer vs pop LIF all | +0.35 | 5/12 | no clear difference | 0.83 | -0.73, +1.53, +0.36, +0.26 | +0.44 | +0.34 |
| C3 pop LIF mixer+mean vs pop LIF mixer | +0.20 | 4/12 | no clear difference | 0.84 | +0.05, +0.54, +0.09, +0.12 | +0.08 | -0.11 |
| C5 pop f-LIF mixer+mean vs pop f-LIF mixer | +0.38 | 2/12 | no clear difference | 0.65 | +1.41, -0.19, +0.13, +0.20 | +0.20 | -0.03 |
| (report) pop f-LIF mixer vs pop LIF mixer | +0.15 | 4/12 | no clear difference | 1.52 | +0.01, -0.03, +0.39, +0.23 | +0.11 | +0.12 |
| (report, mean) pop LIF mixer+mean vs LIF+mean | +0.26 | 3/12 | no clear difference | 1.11 | -0.10, +0.52, +0.23, +0.39 | +0.16 | -0.02 |
| (report) LIF vs SF-de | +0.30 | 5/12 | no clear difference | 0.98 | +1.08, -0.24, +0.42, -0.04 | +0.30 | +0.05 |
| (report) LIF+mean vs SF-de | -0.36 | 7/12 | no clear difference | 1.11 | -0.95, -0.70, +0.30, -0.08 | -0.05 | +0.31 |
| (report) pop LIF all vs SF-de | -0.64 | 5/12 | no clear difference | 1.82 | -0.37, -2.22, +0.08, -0.07 | -0.41 | +0.06 |
| (report) pop LIF mixer vs SF-de | -0.30 | 5/12 | no clear difference | 1.48 | -1.10, -0.72, +0.44, +0.19 | +0.02 | +0.40 |
| (report) pop LIF mixer+mean vs SF-de | -0.10 | 4/12 | no clear difference | 1.23 | -1.05, -0.19, +0.54, +0.31 | +0.10 | +0.29 |
| (report) pop f-LIF mixer vs SF-de | -0.15 | 5/12 | no clear difference | 2.30 | -1.09, -0.77, +0.83, +0.42 | +0.13 | +0.53 |
| (report) pop f-LIF mixer+mean vs SF-de | +0.23 | 3/12 | no clear difference | 1.44 | +0.30, -0.96, +0.96, +0.62 | +0.33 | +0.50 |

Pairs r % (test MSE), seed 42, 7, 2026 per cell (verdict comparisons):

- B1 pop LIF mixer vs LIF: ETTh2 H96 -3.49, +0.02, -2.92; ETTh2 H720 -1.84, -0.28, +0.68; ETTm2 H96 -0.16, -0.24, +0.45; ETTm2 H720 +0.33, +0.22, +0.15
- B3 pop LIF mixer vs pop LIF all: ETTh2 H96 -0.27, -0.10, -1.83; ETTh2 H720 +1.62, +0.98, +1.99; ETTm2 H96 +0.15, -0.23, +1.16; ETTm2 H720 -0.03, +0.36, +0.44
- C3 pop LIF mixer+mean vs pop LIF mixer: ETTh2 H96 +0.53, -0.83, +0.46; ETTh2 H720 +1.37, +1.57, -1.31; ETTm2 H96 +0.57, +0.17, -0.47; ETTm2 H720 +0.35, +0.17, -0.15
- C5 pop f-LIF mixer+mean vs pop f-LIF mixer: ETTh2 H96 +0.45, +0.68, +3.10; ETTh2 H720 +0.53, +0.17, -1.29; ETTm2 H96 +0.38, -0.35, +0.36; ETTm2 H720 +0.03, +0.32, +0.25

## Report only: stopping and firing rates (seed mean)

| cell | arm | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | energy µJ | time s (seeds) |
|---|---|---|---|--:|--:|--:|---|
| ETTh2 H96 | LIF | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 13.51 | 461, 404, 498 |
| ETTh2 H96 | LIF+mean | 0, 2, 0 | 11, 13, 11 | 0.2952 | 0.0709 | 16.23 | 294, 412, 309 |
| ETTh2 H96 | pop LIF all | 0, 1, 1 | 11, 12, 12 | 0.2892 | 0.0715 | 31.07 | 598, 790, 806 |
| ETTh2 H96 | pop LIF mixer | 0, 1, 0 | 11, 12, 11 | 0.2914 | 0.0742 | 24.34 | 486, 506, 492 |
| ETTh2 H96 | pop LIF mixer+mean | 0, 1, 0 | 11, 12, 11 | 0.2902 | 0.0730 | 18.94 | 269, 282, 272 |
| ETTh2 H96 | pop f-LIF mixer | 0, 1, 0 | 11, 12, 11 | 0.2907 | 0.0645 | 42.22 | 401, 428, 500 |
| ETTh2 H96 | pop f-LIF mixer+mean | 0, 1, 3 | 11, 12, 14 | 0.2986 | 0.0636 | 23.47 | 324, 338, 342 |
| ETTh2 H96 | SF-de | 1, 1, 3 | 12, 12, 14 | 0.2894 | 0.0572 | 13.84 | 280, 253, 302 |
| ETTh2 H720 | LIF | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 44.23 | 377, 342, 416 |
| ETTh2 H720 | LIF+mean | 3, 0, 2 | 14, 11, 13 | 0.4228 | 0.0716 | 47.09 | 343, 433, 339 |
| ETTh2 H720 | pop LIF all | 1, 0, 2 | 12, 11, 13 | 0.4201 | 0.0678 | 61.84 | 696, 697, 814 |
| ETTh2 H720 | pop LIF mixer | 1, 3, 2 | 12, 14, 13 | 0.4251 | 0.0735 | 55.11 | 376, 440, 408 |
| ETTh2 H720 | pop LIF mixer+mean | 3, 3, 2 | 14, 14, 13 | 0.4247 | 0.0722 | 49.78 | 353, 351, 400 |
| ETTh2 H720 | pop f-LIF mixer | 1, 0, 2 | 12, 11, 13 | 0.4254 | 0.0626 | 73.00 | 402, 356, 442 |
| ETTh2 H720 | pop f-LIF mixer+mean | 1, 0, 2 | 12, 11, 13 | 0.4197 | 0.0621 | 54.32 | 331, 302, 421 |
| ETTh2 H720 | SF-de | 3, 3, 2 | 14, 14, 13 | 0.4233 | 0.0534 | 44.64 | 301, 309, 329 |
| ETTm2 H96 | LIF | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 14.37 | 3829, 2399, 2162 |
| ETTm2 H96 | LIF+mean | 15, 11, 21 | 26, 22, 32 | 0.1772 | 0.0843 | 16.99 | 3745, 3582, 2872 |
| ETTm2 H96 | pop LIF all | 8, 5, 6 | 19, 16, 17 | 0.1768 | 0.0799 | 33.10 | 3694, 3753, 3355 |
| ETTm2 H96 | pop LIF mixer | 4, 19, 7 | 15, 30, 18 | 0.1774 | 0.0813 | 25.16 | 2495, 4775, 3366 |
| ETTm2 H96 | pop LIF mixer+mean | 6, 9, 10 | 17, 20, 21 | 0.1780 | 0.0821 | 19.67 | 2164, 2478, 2053 |
| ETTm2 H96 | pop f-LIF mixer | 11, 7, 13 | 22, 18, 24 | 0.1782 | 0.0749 | 43.05 | 3951, 3510, 3638 |
| ETTm2 H96 | pop f-LIF mixer+mean | 33, 14, 13 | 44, 25, 24 | 0.1786 | 0.0759 | 24.22 | 6146, 3605, 2426 |
| ETTm2 H96 | SF-de | 15, 11, 12 | 26, 22, 23 | 0.1769 | 0.0727 | 14.80 | 1513, 1276, 1326 |
| ETTm2 H720 | LIF | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 45.08 | 1729, 1905, 1600 |
| ETTm2 H720 | LIF+mean | 2, 3, 2 | 13, 14, 13 | 0.4079 | 0.0729 | 47.77 | 1888, 2065, 1859 |
| ETTm2 H720 | pop LIF all | 2, 1, 1 | 13, 12, 12 | 0.4104 | 0.0705 | 63.84 | 2649, 2528, 2341 |
| ETTm2 H720 | pop LIF mixer | 1, 1, 1 | 12, 12, 12 | 0.4133 | 0.0706 | 55.86 | 1473, 1502, 1571 |
| ETTm2 H720 | pop LIF mixer+mean | 2, 2, 1 | 13, 13, 12 | 0.4133 | 0.0709 | 50.47 | 1561, 1356, 1185 |
| ETTm2 H720 | pop f-LIF mixer | 2, 2, 2 | 13, 13, 13 | 0.4122 | 0.0669 | 73.76 | 1784, 1714, 1708 |
| ETTm2 H720 | pop f-LIF mixer+mean | 2, 4, 2 | 13, 15, 13 | 0.4127 | 0.0652 | 55.01 | 1439, 1914, 1383 |
| ETTm2 H720 | SF-de | 2, 4, 2 | 13, 15, 13 | 0.4091 | 0.0627 | 45.44 | 1051, 1132, 992 |

## Energy A/B under both rules (report)

| comparison | syops rule | old rule (total_fft_uJ_corr) |
|---|--:|--:|
| B1 pop LIF mixer vs LIF | 1.5093 | 2.0625 |
| B3 pop LIF mixer vs pop LIF all | 0.8275 | 0.7410 |
| C3 pop LIF mixer+mean vs pop LIF mixer | 0.8416 | 0.7418 |
| C5 pop f-LIF mixer+mean vs pop f-LIF mixer | 0.6521 | 0.5173 |
| (report) pop f-LIF mixer vs pop LIF mixer | 1.5225 | 1.8416 |
| (report, mean) pop LIF mixer+mean vs LIF+mean | 1.1096 | 1.2058 |
| (report) LIF vs SF-de | 0.9825 | 0.9637 |
| (report) LIF+mean vs SF-de | 1.1065 | 1.2218 |
| (report) pop LIF all vs SF-de | 1.8176 | 2.6829 |
| (report) pop LIF mixer vs SF-de | 1.4805 | 1.9876 |
| (report) pop LIF mixer+mean vs SF-de | 1.2306 | 1.4735 |
| (report) pop f-LIF mixer vs SF-de | 2.3040 | 3.6641 |
| (report) pop f-LIF mixer+mean vs SF-de | 1.4397 | 1.8928 |

## Adoption (user rule 11-19): 4-cell mean test MSE and MAE both >= 0.001 below the LIF SpkMixer, and syops energy A/B vs SF-de <= 1.05

LIF SpkMixer 4-cell mean MSE / MAE: 0.324497 / 0.356692

| arm | 4-cell mean MSE / MAE | LIF − arm MSE / MAE | energy A/B vs SF-de (syops) | MSE ok | MAE ok | energy ok | adopted candidate |
|---|---|---|--:|---|---|---|---|
| pop LIF mixer | 0.3226 / 0.3557 | +0.0019 / +0.0010 | 1.4805 | yes | no | no | no |
| pop LIF mixer+mean | 0.3234 / 0.3561 | +0.0011 / +0.0006 | 1.2306 | yes | no | no | no |
| pop f-LIF mixer | 0.3230 / 0.3561 | +0.0015 / +0.0006 | 2.3040 | yes | no | no | no |
| pop f-LIF mixer+mean | 0.3241 / 0.3567 | +0.0004 / -0.0000 | 1.4397 | no | no | no | no |

**Adopted structure for the tuning: LIF** (no arm passes the rule, so the LIF SpkMixer stays).

## User pass criteria (report; results/ett-pass-20261119/pass_ett.py)

### LIF

4-cell mean MSE / MAE: **0.3245 / 0.3567**; energy A/B vs SF-de (syops) 0.9825

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | -0.0030 / -0.0017 | no |
| SF-de (ours) | 0.3238 / 0.3557 | -0.0007 / -0.0010 | no |
| energy A/B vs SF-de (syops; bound 1.05) | 0.9825 | | yes |
| iTransformer | 0.3290 / 0.3663 | +0.0045 / +0.0096 | yes |
| iTransformer, MAE loss | 0.3275 / 0.3623 | +0.0030 / +0.0056 | yes |
| RLinear | 0.3242 / 0.3603 | -0.0002 / +0.0036 | no |
| PatchTST (re-run) | 0.3275 / 0.3632 | +0.0030 / +0.0066 | yes |
| PatchTST (other re-run) | 0.3347 / 0.3698 | +0.0103 / +0.0131 | yes |
| TimeMixer | 0.3167 / 0.3572 | -0.0077 / +0.0006 | no |
| TimesNet | 0.3493 / 0.3780 | +0.0248 / +0.0213 | yes |
| FITS | 0.3290 / 0.3653 | +0.0045 / +0.0086 | yes |
| DLinear (re-run) | 0.4778 / 0.4645 | +0.1533 / +0.1078 | yes |

- (1) SNN: **fail**; (2) Spikformer: **fail**; (3) DNN (all 9 rows): **fail** (7/9 rows); overall: **NOT PASSED**
- Needed for all three: 4-cell mean MSE ≤ 0.3157 (now 0.3245, short by 0.0087) and MAE ≤ 0.3540 (now 0.3567, short by 0.0027); plus energy A/B ≤ 1.05.
- Report (not a criterion): NSMT 0.3177 / 0.3518, ref − ours -0.0067 / -0.0049.

### pop LIF mixer

4-cell mean MSE / MAE: **0.3226 / 0.3557**; energy A/B vs SF-de (syops) 1.4805

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | -0.0011 / -0.0007 | no |
| SF-de (ours) | 0.3238 / 0.3557 | +0.0012 / +0.0000 | no |
| energy A/B vs SF-de (syops; bound 1.05) | 1.4805 | | no |
| iTransformer | 0.3290 / 0.3663 | +0.0064 / +0.0105 | yes |
| iTransformer, MAE loss | 0.3275 / 0.3623 | +0.0049 / +0.0065 | yes |
| RLinear | 0.3242 / 0.3603 | +0.0016 / +0.0045 | yes |
| PatchTST (re-run) | 0.3275 / 0.3632 | +0.0049 / +0.0075 | yes |
| PatchTST (other re-run) | 0.3347 / 0.3698 | +0.0121 / +0.0140 | yes |
| TimeMixer | 0.3167 / 0.3572 | -0.0059 / +0.0015 | no |
| TimesNet | 0.3493 / 0.3780 | +0.0266 / +0.0223 | yes |
| FITS | 0.3290 / 0.3653 | +0.0064 / +0.0095 | yes |
| DLinear (re-run) | 0.4778 / 0.4645 | +0.1551 / +0.1088 | yes |

- (1) SNN: **fail**; (2) Spikformer: **fail**; (3) DNN (all 9 rows): **fail** (8/9 rows); overall: **NOT PASSED**
- Needed for all three: 4-cell mean MSE ≤ 0.3157 (now 0.3226, short by 0.0069) and MAE ≤ 0.3540 (now 0.3557, short by 0.0017); plus energy A/B ≤ 1.05.
- Report (not a criterion): NSMT 0.3177 / 0.3518, ref − ours -0.0049 / -0.0040.

### pop LIF mixer+mean

4-cell mean MSE / MAE: **0.3234 / 0.3561**; energy A/B vs SF-de (syops) 1.2306

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | -0.0019 / -0.0011 | no |
| SF-de (ours) | 0.3238 / 0.3557 | +0.0004 / -0.0003 | no |
| energy A/B vs SF-de (syops; bound 1.05) | 1.2306 | | no |
| iTransformer | 0.3290 / 0.3663 | +0.0056 / +0.0102 | yes |
| iTransformer, MAE loss | 0.3275 / 0.3623 | +0.0041 / +0.0062 | yes |
| RLinear | 0.3242 / 0.3603 | +0.0008 / +0.0042 | no |
| PatchTST (re-run) | 0.3275 / 0.3632 | +0.0041 / +0.0072 | yes |
| PatchTST (other re-run) | 0.3347 / 0.3698 | +0.0113 / +0.0137 | yes |
| TimeMixer | 0.3167 / 0.3572 | -0.0067 / +0.0012 | no |
| TimesNet | 0.3493 / 0.3780 | +0.0258 / +0.0219 | yes |
| FITS | 0.3290 / 0.3653 | +0.0056 / +0.0092 | yes |
| DLinear (re-run) | 0.4778 / 0.4645 | +0.1543 / +0.1084 | yes |

- (1) SNN: **fail**; (2) Spikformer: **fail**; (3) DNN (all 9 rows): **fail** (7/9 rows); overall: **NOT PASSED**
- Needed for all three: 4-cell mean MSE ≤ 0.3157 (now 0.3234, short by 0.0077) and MAE ≤ 0.3540 (now 0.3561, short by 0.0021); plus energy A/B ≤ 1.05.
- Report (not a criterion): NSMT 0.3177 / 0.3518, ref − ours -0.0057 / -0.0043.

### pop f-LIF mixer

4-cell mean MSE / MAE: **0.3230 / 0.3561**; energy A/B vs SF-de (syops) 2.3040

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | -0.0015 / -0.0011 | no |
| SF-de (ours) | 0.3238 / 0.3557 | +0.0008 / -0.0004 | no |
| energy A/B vs SF-de (syops; bound 1.05) | 2.3040 | | no |
| iTransformer | 0.3290 / 0.3663 | +0.0060 / +0.0102 | yes |
| iTransformer, MAE loss | 0.3275 / 0.3623 | +0.0045 / +0.0062 | yes |
| RLinear | 0.3242 / 0.3603 | +0.0012 / +0.0042 | yes |
| PatchTST (re-run) | 0.3275 / 0.3632 | +0.0045 / +0.0072 | yes |
| PatchTST (other re-run) | 0.3347 / 0.3698 | +0.0117 / +0.0137 | yes |
| TimeMixer | 0.3167 / 0.3572 | -0.0063 / +0.0012 | no |
| TimesNet | 0.3493 / 0.3780 | +0.0262 / +0.0219 | yes |
| FITS | 0.3290 / 0.3653 | +0.0060 / +0.0092 | yes |
| DLinear (re-run) | 0.4778 / 0.4645 | +0.1547 / +0.1084 | yes |

- (1) SNN: **fail**; (2) Spikformer: **fail**; (3) DNN (all 9 rows): **fail** (8/9 rows); overall: **NOT PASSED**
- Needed for all three: 4-cell mean MSE ≤ 0.3157 (now 0.3230, short by 0.0073) and MAE ≤ 0.3540 (now 0.3561, short by 0.0021); plus energy A/B ≤ 1.05.
- Report (not a criterion): NSMT 0.3177 / 0.3518, ref − ours -0.0053 / -0.0043.

### pop f-LIF mixer+mean

4-cell mean MSE / MAE: **0.3241 / 0.3567**; energy A/B vs SF-de (syops) 1.4397

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | -0.0026 / -0.0017 | no |
| SF-de (ours) | 0.3238 / 0.3557 | -0.0002 / -0.0010 | no |
| energy A/B vs SF-de (syops; bound 1.05) | 1.4397 | | no |
| iTransformer | 0.3290 / 0.3663 | +0.0049 / +0.0095 | yes |
| iTransformer, MAE loss | 0.3275 / 0.3623 | +0.0034 / +0.0055 | yes |
| RLinear | 0.3242 / 0.3603 | +0.0002 / +0.0035 | no |
| PatchTST (re-run) | 0.3275 / 0.3632 | +0.0034 / +0.0065 | yes |
| PatchTST (other re-run) | 0.3347 / 0.3698 | +0.0107 / +0.0130 | yes |
| TimeMixer | 0.3167 / 0.3572 | -0.0073 / +0.0005 | no |
| TimesNet | 0.3493 / 0.3780 | +0.0252 / +0.0213 | yes |
| FITS | 0.3290 / 0.3653 | +0.0049 / +0.0085 | yes |
| DLinear (re-run) | 0.4778 / 0.4645 | +0.1537 / +0.1078 | yes |

- (1) SNN: **fail**; (2) Spikformer: **fail**; (3) DNN (all 9 rows): **fail** (7/9 rows); overall: **NOT PASSED**
- Needed for all three: 4-cell mean MSE ≤ 0.3157 (now 0.3241, short by 0.0083) and MAE ≤ 0.3540 (now 0.3567, short by 0.0027); plus energy A/B ≤ 1.05.
- Report (not a criterion): NSMT 0.3177 / 0.3518, ref − ours -0.0063 / -0.0050.

