# SpikF head activation: GELU (released) vs LIF, input 96, ETT 16 cells

Rules: summarize_spikf_act.py docstring. 96 runs, evaluation setting `1_fe256_sy1`, training commit `ff425911c`. Ours = seed mean of 42 / 7 / 2026 (best-validation weights). Energy = syops count with 0.9 pJ (AC) / 4.6 pJ (MAC), as SpikF logs it.

## 16-cell means

| model | MSE | MAE | energy µJ |
|---|--:|--:|--:|
| SpikF paper (single published numbers) | 0.3700 | 0.3819 | — |
| SpikF-GELU (these runs) | 0.3701 | 0.3818 | 908.02 |
| SpikF-LIF (these runs) | 0.3659 | 0.3796 | 785.50 |
| ours: first chain (11-25) | 0.3716 | 0.3815 | 11.11 |
| ours: extension (11-25) | 0.3724 | 0.3826 | 24.09 |

## A. Reproduction: SpikF-GELU (these runs) − paper, per dataset (mean of 4 horizons; + = ours higher)

| dataset | MSE | MAE |
|---|--:|--:|
| ETTh1 | +0.0016 | +0.0001 |
| ETTh2 | +0.0004 | -0.0001 |
| ETTm1 | -0.0009 | -0.0001 |
| ETTm2 | -0.0007 | -0.0001 |
| 16 cells | +0.0001 | -0.0000 |

## B. LIF vs GELU (paired by cell and seed)

- test MSE: R -1.14 %, pairs higher 5 / lower 43 of 48 -> **LIF better**
- test MAE: R -0.59 %, pairs higher 6 / lower 42 of 48 -> **no clear difference**
- energy LIF / GELU: mean of cell ratios 0.8726

| cell | GELU MSE / MAE | LIF MSE / MAE | r MSE % | GELU µJ | LIF µJ | LIF / GELU | epochs run G / L |
|---|---|---|--:|--:|--:|--:|---|
| ETTh1 H96 | 0.3823 / 0.3912 | 0.3756 / 0.3888 | -1.75 | 819.23 | 784.06 | 0.9571 | [5, 5, 5] / [5, 5, 5] |
| ETTh1 H192 | 0.4339 / 0.4213 | 0.4295 / 0.4199 | -1.02 | 854.81 | 784.77 | 0.9181 | [5, 5, 5] / [5, 5, 5] |
| ETTh1 H336 | 0.4765 / 0.4417 | 0.4737 / 0.4405 | -0.57 | 908.22 | 785.78 | 0.8652 | [5, 5, 5] / [5, 5, 5] |
| ETTh1 H720 | 0.4716 / 0.4582 | 0.4719 / 0.4571 | +0.06 | 1050.55 | 788.33 | 0.7504 | [5, 4, 5] / [5, 5, 5] |
| ETTh2 H96 | 0.2905 / 0.3355 | 0.2861 / 0.3311 | -1.51 | 819.05 | 783.77 | 0.9569 | [4, 5, 4] / [4, 5, 4] |
| ETTh2 H192 | 0.3678 / 0.3849 | 0.3626 / 0.3822 | -1.40 | 854.61 | 784.42 | 0.9179 | [5, 3, 3] / [5, 5, 5] |
| ETTh2 H336 | 0.4122 / 0.4205 | 0.4076 / 0.4179 | -1.09 | 908.03 | 785.14 | 0.8647 | [4, 4, 4] / [4, 4, 4] |
| ETTh2 H720 | 0.4183 / 0.4358 | 0.4177 / 0.4360 | -0.15 | 1050.44 | 787.03 | 0.7492 | [3, 3, 4] / [3, 3, 3] |
| ETTm1 H96 | 0.3177 / 0.3445 | 0.3124 / 0.3413 | -1.69 | 819.12 | 784.01 | 0.9571 | [3, 3, 3] / [3, 3, 3] |
| ETTm1 H192 | 0.3717 / 0.3728 | 0.3635 / 0.3681 | -2.21 | 854.69 | 784.92 | 0.9184 | [5, 4, 4] / [5, 4, 5] |
| ETTm1 H336 | 0.3976 / 0.3929 | 0.3909 / 0.3903 | -1.69 | 908.02 | 786.09 | 0.8657 | [3, 5, 4] / [5, 6, 4] |
| ETTm1 H720 | 0.4602 / 0.4304 | 0.4502 / 0.4264 | -2.15 | 1050.54 | 788.90 | 0.7510 | [5, 6, 5] / [4, 4, 5] |
| ETTm2 H96 | 0.1749 / 0.2524 | 0.1728 / 0.2507 | -1.19 | 818.76 | 783.74 | 0.9572 | [2, 3, 4] / [2, 3, 4] |
| ETTm2 H192 | 0.2399 / 0.2950 | 0.2381 / 0.2941 | -0.73 | 854.34 | 784.35 | 0.9181 | [3, 4, 4] / [3, 3, 3] |
| ETTm2 H336 | 0.3010 / 0.3356 | 0.2993 / 0.3342 | -0.59 | 907.83 | 785.38 | 0.8651 | [2, 4, 3] / [2, 4, 3] |
| ETTm2 H720 | 0.4053 / 0.3965 | 0.4030 / 0.3956 | -0.56 | 1050.12 | 787.31 | 0.7497 | [3, 2, 3] / [2, 2, 3] |

## C. Our models against SpikF-LIF (and SpikF-GELU) of these runs (margin = ref − ours, ≥ +0.001 needed for both)

| our model | reference | margin MSE / MAE | accuracy | energy cells below | verdict |
|---|---|---|---|--:|---|
| first chain | SpikF-LIF | -0.0057 / -0.0019 | no | 16 / 16 | **NOT PASSED** |
| first chain | SpikF-GELU | -0.0015 / +0.0003 | no | 16 / 16 | **NOT PASSED** |
| extension | SpikF-LIF | -0.0064 / -0.0029 | no | 16 / 16 | **NOT PASSED** |
| extension | SpikF-GELU | -0.0023 / -0.0007 | no | 16 / 16 | **NOT PASSED** |

Seed spread (std of test MSE over 3 seeds, mean over 16 cells): GELU 0.0027, LIF 0.0028
