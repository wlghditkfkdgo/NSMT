# SF-de with patch 16 (fair-patch comparison; test values, seeds 42 / 7 / 2026, input 96)

Rules and sources: module docstring of summarize_sfpt16.py.

## Test MSE / MAE (seed mean) and energy (µJ per input window)

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | 4-cell MSE / MAE | energy syops+neuron | energy syops only | params ETTh2 H96 / H720 |
|---|---|---|---|---|---|--:|--:|---|
| first-chain final (LIF SpkMixer, patch 16) | 0.2902 / 0.3347 | 0.4151 / 0.4327 | 0.1748 / 0.2525 | 0.4018 / 0.3940 | 0.3205 / 0.3535 | 15.75 | 13.94 | 96064 / 575296 |
| extension final (LIF SpkMixer, patch 24) | 0.2887 / 0.3327 | 0.4141 / 0.4318 | 0.1745 / 0.2521 | 0.3999 / 0.3937 | 0.3193 / 0.3526 | 36.10 | 31.26 | 72000 / 391488 |
| SF-de patch 16 | 0.2866 / 0.3320 | 0.4175 / 0.4334 | 0.1750 / 0.2525 | 0.4015 / 0.3942 | 0.3201 / 0.3530 | 15.68 | 13.74 | 96248 / 575480 |
| SF-de patch 8 (criteria SF-de) | 0.2921 / 0.3360 | 0.4229 / 0.4377 | 0.1768 / 0.2538 | 0.4036 / 0.3953 | 0.3238 / 0.3557 | 29.68 | 25.81 | 169656 / 1128120 |

Energy per cell (seed mean, µJ; syops+neuron / syops only):

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|--:|--:|--:|--:|
| first-chain final (LIF SpkMixer, patch 16) | 7.54 / 5.92 | 22.93 / 21.31 | 8.58 / 6.58 | 23.94 / 21.94 |
| extension final (LIF SpkMixer, patch 24) | 5.58 / 4.50 | 15.84 / 14.76 | 41.06 / 32.44 | 81.93 / 73.32 |
| SF-de patch 16 | 7.79 / 5.91 | 23.19 / 21.32 | 8.19 / 6.17 | 23.55 / 21.55 |
| SF-de patch 8 (criteria SF-de) | 13.84 / 10.11 | 44.64 / 40.92 | 14.80 / 10.77 | 45.44 / 41.43 |

## Comparisons (ETT rule; test MSE)

| comparison | R % | lower pairs | verdict | per cell r % | MAE R % | energy A/B syops+neuron | energy A/B syops only |
|---|--:|--:|---|---|--:|--:|--:|
| **main** first-chain final vs SF-de patch 16 | +0.16 | 6/12 | no clear difference | +1.27, -0.57, -0.15, +0.08 | +0.15 | 1.0052 | 1.0212 |
| (report) extension final vs SF-de patch 16 | -0.19 | 8/12 | no clear difference | +0.75, -0.81, -0.31, -0.38 | -0.11 | 2.4730 | 2.5282 |
| (report) SF-de patch 16 vs SF-de patch 8 | -1.17 | 12/12 | better | -1.89, -1.26, -1.00, -0.54 | -0.74 | 0.5384 | 0.5497 |
| (report) first-chain final vs SF-de patch 8 | -1.02 | 11/12 | better | -0.65, -1.82, -1.14, -0.46 | -0.59 | 0.5412 | 0.5616 |

## Pass criteria (report; NSMT-paper DNN rows; pass_ett_nsmt.judge)

#### first-chain final (LIF SpkMixer, patch 16); criterion (2) against SF-de patch 8 (as defined)

4-cell mean MSE / MAE: **0.3205 / 0.3535**; energy A/B vs SF-de (syops) 0.5412

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | +0.0010 / +0.0015 | yes |
| SF-de (ours) | 0.3238 / 0.3557 | +0.0034 / +0.0022 | yes |
| energy A/B vs SF-de (syops; bound 1.05) | 0.5412 | | yes |
| Informer | 2.7865 / 1.2352 | +2.4660 / +0.8818 | yes |
| FEDformer | 0.3613 / 0.3932 | +0.0408 / +0.0398 | yes |
| PatchTST | 0.3275 / 0.3632 | +0.0070 / +0.0098 | yes |
| ETSformer | 0.3608 / 0.3953 | +0.0403 / +0.0418 | yes |
| Autoformer | 0.3873 / 0.4175 | +0.0668 / +0.0640 | yes |

- (1) SNN: **pass**; (2) Spikformer: **pass**; (3) DNN (5 rows): **pass** (5/5); overall: **PASS**

#### first-chain final (LIF SpkMixer, patch 16); criterion (2) against SF-de patch 16 (if the criterion used it)

4-cell mean MSE / MAE: **0.3205 / 0.3535**; energy A/B vs SF-de (syops) 1.0052

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | +0.0010 / +0.0015 | yes |
| SF-de (ours) | 0.3201 / 0.3530 | -0.0003 / -0.0004 | no |
| energy A/B vs SF-de (syops; bound 1.05) | 1.0052 | | yes |
| Informer | 2.7865 / 1.2352 | +2.4660 / +0.8818 | yes |
| FEDformer | 0.3613 / 0.3932 | +0.0408 / +0.0398 | yes |
| PatchTST | 0.3275 / 0.3632 | +0.0070 / +0.0098 | yes |
| ETSformer | 0.3608 / 0.3953 | +0.0403 / +0.0418 | yes |
| Autoformer | 0.3873 / 0.4175 | +0.0668 / +0.0640 | yes |

- (1) SNN: **pass**; (2) Spikformer: **fail**; (3) DNN (5 rows): **pass** (5/5); overall: **NOT PASSED**

#### extension final (LIF SpkMixer, patch 24); criterion (2) against SF-de patch 8 (as defined)

4-cell mean MSE / MAE: **0.3193 / 0.3526**; energy A/B vs SF-de (syops) 1.3338

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | +0.0022 / +0.0024 | yes |
| SF-de (ours) | 0.3238 / 0.3557 | +0.0045 / +0.0031 | yes |
| energy A/B vs SF-de (syops; bound 1.05) | 1.3338 | | no |
| Informer | 2.7865 / 1.2352 | +2.4672 / +0.8827 | yes |
| FEDformer | 0.3613 / 0.3932 | +0.0419 / +0.0407 | yes |
| PatchTST | 0.3275 / 0.3632 | +0.0082 / +0.0107 | yes |
| ETSformer | 0.3608 / 0.3953 | +0.0414 / +0.0427 | yes |
| Autoformer | 0.3873 / 0.4175 | +0.0679 / +0.0649 | yes |

- (1) SNN: **pass**; (2) Spikformer: **fail**; (3) DNN (5 rows): **pass** (5/5); overall: **NOT PASSED**

#### extension final (LIF SpkMixer, patch 24); criterion (2) against SF-de patch 16 (if the criterion used it)

4-cell mean MSE / MAE: **0.3193 / 0.3526**; energy A/B vs SF-de (syops) 2.4730

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | +0.0022 / +0.0024 | yes |
| SF-de (ours) | 0.3201 / 0.3530 | +0.0008 / +0.0005 | no |
| energy A/B vs SF-de (syops; bound 1.05) | 2.4730 | | no |
| Informer | 2.7865 / 1.2352 | +2.4672 / +0.8827 | yes |
| FEDformer | 0.3613 / 0.3932 | +0.0419 / +0.0407 | yes |
| PatchTST | 0.3275 / 0.3632 | +0.0082 / +0.0107 | yes |
| ETSformer | 0.3608 / 0.3953 | +0.0414 / +0.0427 | yes |
| Autoformer | 0.3873 / 0.4175 | +0.0679 / +0.0649 | yes |

- (1) SNN: **pass**; (2) Spikformer: **fail**; (3) DNN (5 rows): **pass** (5/5); overall: **NOT PASSED**

