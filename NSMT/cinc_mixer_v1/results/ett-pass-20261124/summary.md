# SNN models at look-back 96 (SpikF paper rows) vs ours; energy = syops only (user 11-24)

Rules and sources: module docstring of snn_lb96_syops.py. The SpikF paper has no plain Spikformer row at input 96; its Spikformer-family baseline is iSpikformer (with SPE).

## Test MSE / MAE and energy

| model | source | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | 4-cell MSE / MAE | energy syops µJ (4-cell mean; H720 ETTh2 / ETTm2) |
|---|---|---|---|---|---|---|---|
| NSMT | NSMT Table 14 (paper value) | 0.2880 / 0.3340 | 0.4090 / 0.4290 | 0.1750 / 0.2510 | 0.3990 / 0.3930 | 0.3177 / 0.3518 | 40.87 (44.90 / 46.72) |
| LIF SpkMixer extension final (patch 24) | ours, 3 seeds | 0.2887 / 0.3327 | 0.4141 / 0.4318 | 0.1745 / 0.2521 | 0.3999 / 0.3937 | 0.3193 / 0.3526 | 31.26 (14.76 / 73.32) |
| SF-de patch 16 (ours, Spikformer-style) | ours, 3 seeds | 0.2866 / 0.3320 | 0.4175 / 0.4334 | 0.1750 / 0.2525 | 0.4015 / 0.3942 | 0.3201 / 0.3530 | 13.74 (21.32 / 21.55) |
| LIF SpkMixer first-chain final (patch 16) | ours, 3 seeds | 0.2902 / 0.3347 | 0.4151 / 0.4327 | 0.1748 / 0.2525 | 0.4018 / 0.3940 | 0.3205 / 0.3535 | 13.94 (21.31 / 21.94) |
| SpikF | SpikF Table 12 (paper value) | 0.2900 / 0.3360 | 0.4160 / 0.4360 | 0.1750 / 0.2510 | 0.4050 / 0.3970 | 0.3215 / 0.3550 | 934.56 (1050.47 / 1050.10) |
| SF-de patch 8 (ours, criteria SF-de) | ours, 3 seeds | 0.2921 / 0.3360 | 0.4229 / 0.4377 | 0.1768 / 0.2538 | 0.4036 / 0.3953 | 0.3238 / 0.3557 | 25.81 (40.92 / 41.43) |
| SpikeRNN | SpikF Table 12 (paper value) | 0.2950 / 0.3350 | 0.4280 / 0.4380 | 0.1760 / 0.2510 | 0.4090 / 0.3960 | 0.3270 / 0.3550 | not available |
| iSpikformer | SpikF Table 12 (paper value) | 0.3080 / 0.3490 | 0.4380 / 0.4390 | 0.1780 / 0.2560 | 0.4060 / 0.3980 | 0.3325 / 0.3605 | not available |

## First-chain final vs each paper SNN row (negative = ours lower = better)

| vs | 4-cell MSE diff (ref − ours) | 4-cell MAE diff | MSE % per cell | cells lower MSE / MAE | 0.001 rule (MSE and MAE) | energy ratio ours / ref (syops) |
|---|--:|--:|---|---|---|---|
| NSMT | -0.0027 | -0.0017 | +0.76, +1.50, -0.13, +0.70 | 1/4 / 0/4 | does not pass | 0.323 (cells 0.177, 0.475, 0.171, 0.470) |
| SpikF | +0.0010 | +0.0015 | +0.06, -0.21, -0.13, -0.79 | 3/4 / 3/4 | passes | 0.014 (cells 0.007, 0.020, 0.008, 0.021) |
| SpikeRNN | +0.0065 | +0.0015 | -1.63, -3.01, -0.70, -1.76 | 4/4 / 3/4 | passes | not available |
| iSpikformer | +0.0120 | +0.0070 | -5.79, -5.22, -1.81, -1.04 | 4/4 / 4/4 | passes | not available |

## Pass criteria re-judged with syops-only energy (NSMT DNN rows; pass_ett_nsmt.judge)

#### LIF SpkMixer first-chain final (patch 16); (2) against SF-de patch 8, energy syops only

4-cell mean MSE / MAE: **0.3205 / 0.3535**; energy A/B vs SF-de (syops) 0.5616

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | +0.0010 / +0.0015 | yes |
| SF-de (ours) | 0.3238 / 0.3557 | +0.0034 / +0.0022 | yes |
| energy A/B vs SF-de (syops; bound 1.05) | 0.5616 | | yes |
| Informer | 2.7865 / 1.2352 | +2.4660 / +0.8818 | yes |
| FEDformer | 0.3613 / 0.3932 | +0.0408 / +0.0398 | yes |
| PatchTST | 0.3275 / 0.3632 | +0.0070 / +0.0098 | yes |
| ETSformer | 0.3608 / 0.3953 | +0.0403 / +0.0418 | yes |
| Autoformer | 0.3873 / 0.4175 | +0.0668 / +0.0640 | yes |

- (1) SNN: **pass**; (2) Spikformer: **pass**; (3) DNN (5 rows): **pass** (5/5); overall: **PASS**

#### LIF SpkMixer extension final (patch 24); (2) against SF-de patch 8, energy syops only

4-cell mean MSE / MAE: **0.3193 / 0.3526**; energy A/B vs SF-de (syops) 1.3971

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | +0.0022 / +0.0024 | yes |
| SF-de (ours) | 0.3238 / 0.3557 | +0.0045 / +0.0031 | yes |
| energy A/B vs SF-de (syops; bound 1.05) | 1.3971 | | no |
| Informer | 2.7865 / 1.2352 | +2.4672 / +0.8827 | yes |
| FEDformer | 0.3613 / 0.3932 | +0.0419 / +0.0407 | yes |
| PatchTST | 0.3275 / 0.3632 | +0.0082 / +0.0107 | yes |
| ETSformer | 0.3608 / 0.3953 | +0.0414 / +0.0427 | yes |
| Autoformer | 0.3873 / 0.4175 | +0.0679 / +0.0649 | yes |

- (1) SNN: **pass**; (2) Spikformer: **fail**; (3) DNN (5 rows): **pass** (5/5); overall: **NOT PASSED**

