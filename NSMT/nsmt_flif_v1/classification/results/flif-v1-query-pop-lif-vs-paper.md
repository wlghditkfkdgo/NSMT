# [query pop LIF NSMT T=4] next to the NSMT paper and Spikformer-style

See the module docstring for sources and rules. Checks passed: seed 42 of the NSMT reproduction equals Table 13 (4 cells) and Table 6 (H720), and the earlier ETT verdicts (query vs Spikformer-style, H6 vs flat) are reproduced.

## Versions of the model that exist (ETT)

| version | cells | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean validation L1 vs default |
|---|---|---|---|---|---|---|
| default: flat head, lr 5e-4, soma reset subtract (test MSE) | 4 | 0.2912 | 0.4135 | 0.1760 | 0.4028 | 1.0000 |
| H6: linear shortcut + small head | 4 | 0.2861 | 0.4172 | 0.1761 | 0.4022 | 1.0000 |
| lr 2.5e-4 | ETTh2 only | 0.2883 | 0.4160 | not run | not run | 1.0006 |
| lr 1e-3 | ETTh2 only | 0.2925 | 0.4179 | not run | not run | 1.0014 |
| lr 2e-3 | ETTh2 only | 0.2987 | 0.4221 | not run | not run | 1.0049 |
| reset zero | ETTh2 only | 0.2931 | 0.4159 | not run | not run | 0.9993 |

Pre-registered selections on validation (ETTh2 two cells): lr 0.0005, reset zero. Per-cell validation L1 of default / H6: ETTh2 H96 0.312312 / 0.310943; ETTh2 H720 0.536678 / 0.537987; ETTm2 H96 0.237016 / 0.237806; ETTm2 H720 0.361002 / 0.360484.

## ETT test MSE / MAE (mean ± sd of seeds 42, 7, 2026 unless noted; lower is better)

| row | stopping rule | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|---|
| **query pop LIF (default)** MSE | patience 10, best-validation weights | 0.2912 ± 0.0045 | 0.4135 ± 0.0014 | 0.1760 ± 0.0002 | 0.4028 ± 0.0022 |
| **query pop LIF (default)** MAE | | 0.3347 ± 0.0021 | 0.4322 ± 0.0007 | 0.2529 ± 0.0004 | 0.3956 ± 0.0016 |
| query pop LIF, H6 head MSE | patience 10, best-validation weights | 0.2861 ± 0.0036 | 0.4172 ± 0.0024 | 0.1761 ± 0.0004 | 0.4022 ± 0.0001 |
| query pop LIF, H6 head MAE | | 0.3319 ± 0.0021 | 0.4335 ± 0.0016 | 0.2528 ± 0.0002 | 0.3945 ± 0.0007 |
| NSMT reproduction, same rule MSE | patience 10, best-validation weights | 0.2896 ± 0.0012 | 0.4153 ± 0.0039 | 0.1744 ± 0.0001 | 0.4014 ± 0.0006 |
| NSMT reproduction, same rule MAE | | 0.3345 ± 0.0006 | 0.4358 ± 0.0022 | 0.2503 ± 0.0002 | 0.3931 ± 0.0002 |
| NSMT reproduction, paper rule MSE | patience 2, last weights | 0.2835 ± 0.0036 | 0.4097 ± 0.0008 | 0.1750 ± 0.0003 | 0.4002 ± 0.0011 |
| NSMT reproduction, paper rule MAE | | 0.3312 ± 0.0025 | 0.4299 ± 0.0005 | 0.2514 ± 0.0002 | 0.3932 ± 0.0006 |
| Spikformer-style MSE | patience 10, best-validation weights | 0.2903 ± 0.0012 | 0.4223 ± 0.0024 | 0.1775 ± 0.0007 | 0.4034 ± 0.0016 |
| Spikformer-style MAE | | 0.3350 ± 0.0018 | 0.4366 ± 0.0018 | 0.2536 ± 0.0005 | 0.3959 ± 0.0005 |
| query pop LIF (default), seed 42 only, MSE / MAE | patience 10, best-validation weights | 0.2867 / 0.3326 | 0.4146 / 0.4330 | 0.1760 / 0.2525 | 0.4005 / 0.3943 |
| NSMT paper Table 13, MSE / MAE | seed 42, as printed | 0.288 / 0.334 | 0.409 / 0.429 | 0.175 / 0.251 | 0.399 / 0.393 |
| iSpikformer, paper Table 14, MSE / MAE | seed 42, as printed | 0.308 / 0.349 | 0.438 / 0.439 | 0.178 / 0.256 | 0.406 / 0.398 |
| SpikeRNN, paper Table 14, MSE / MAE | seed 42, as printed | 0.295 / 0.335 | 0.428 / 0.438 | 0.176 / 0.251 | 0.409 / 0.396 |
| SpikF (Spiking Fourier Network, not Spikformer), paper Table 14, MSE / MAE | seed 42, as printed | 0.290 / 0.336 | 0.416 / 0.436 | 0.175 / 0.251 | 0.405 / 0.397 |

## query pop LIF (default) vs the others: r % of test MSE (negative = query lower); per cell mean and verdict

| vs | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs lower | verdict | energy A/B |
|---|---|---|---|---|---|---|---|---|
| NSMT reproduction, same rule | +0.58 (no clear difference) | -0.42 (no clear difference) | +0.95 (no clear difference) | +0.34 (no clear difference) | +0.36 | 4/12 | no clear difference | 0.20 |
| NSMT reproduction, paper rule (different rule) | +2.76 (worse) | +0.93 (no clear difference) | +0.58 (no clear difference) | +0.65 (no clear difference) | +1.23 | 1/12 | worse | 0.20 |
| Spikformer-style | +0.30 (no clear difference) | -2.07 (better) | -0.80 (no clear difference) | -0.16 (no clear difference) | -0.68 | 9/12 | no clear difference | 1.24 |
| (H6 vs query default) | -1.75 (better) | +0.88 (no clear difference) | +0.03 (no clear difference) | -0.15 (no clear difference) | -0.25 | 7/12 | no clear difference | 1.20 |
| (H6 vs NSMT reproduction, same rule) | -1.20 (better) | +0.46 (no clear difference) | +0.98 (no clear difference) | +0.19 (no clear difference) | +0.11 | 3/12 | no clear difference | 0.24 |
| (H6 vs Spikformer-style) | -1.47 (better) | -1.21 (better) | -0.77 (no clear difference) | -0.31 (no clear difference) | -0.94 | 11/12 | no clear difference | 1.48 |

Against the printed Table 13 values (r % of MSE; rounding of the printed value is up to 0.3 % at H96 of ETTm2):

| | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean |
|---|---|---|---|---|---|
| 3-seed mean vs Table 13 | +1.11 | +1.11 | +0.60 | +0.95 | +0.94 |
| seed 42 vs Table 13 (same seed) | -0.45 | +1.36 | +0.59 | +0.39 | +0.47 |

## ETT energy (uJ, mean of 3 seeds) and parameters

| row | quantity | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|---|
| query pop LIF (default) | synapses + neurons | 11.27 | 12.92 | 13.23 | 14.36 |
| query pop LIF (default) | synapses only | 5.99 | 7.63 | 6.81 | 8.03 |
| query pop LIF (default) | parameters | 175,544 | 1,134,008 | 184,560 | 1,143,024 |
| query pop LIF, H6 head | synapses + neurons | 11.97 | 17.69 | 13.73 | 19.11 |
| query pop LIF, H6 head | synapses only | 6.68 | 12.42 | 7.33 | 12.81 |
| query pop LIF, H6 head | parameters | 56,248 | 235,960 | 65,264 | 244,976 |
| NSMT reproduction, same rule | synapses + neurons | 56.78 | 68.57 | 64.93 | 71.40 |
| NSMT reproduction, same rule | synapses only | 33.77 | 45.59 | 38.94 | 46.63 |
| NSMT reproduction, same rule | parameters | 178,880 | 1,137,344 | 187,264 | 1,145,728 |
| NSMT reproduction, paper rule | synapses + neurons | 56.00 | 66.89 | 63.79 | 71.62 |
| NSMT reproduction, paper rule | synapses only | 33.17 | 44.09 | 38.24 | 46.77 |
| NSMT reproduction, paper rule | parameters | 178,880 | 1,137,344 | 187,264 | 1,145,728 |
| Spikformer-style | synapses + neurons | 9.30 | 10.79 | 10.28 | 11.48 |
| Spikformer-style | synapses only | 5.58 | 7.08 | 6.23 | 7.52 |
| Spikformer-style | parameters | 169,656 | 1,128,120 | 174,192 | 1,132,656 |
| NSMT paper Table 6 | synapses only | not printed | 44.90 | not printed | 46.72 |

## MIT-BIH test (seeds 42, 7, 2026, last-epoch weights, 30 epochs, lr 0.01)

| row | accuracy % | macro-F1 % | F1 N / S / V / F / Q | synapses uJ | synapses + neurons uJ | parameters |
|---|---|---|---|---|---|---|
| **query pop LIF (default)** | 98.57 ± 0.04 | 92.17 ± 0.16 | 99.26 / 86.18 / 96.08 / 80.36 / 98.98 | 8.47 | 11.10 | 1,230,888 |
| query pop LIF, H2 head | 98.53 ± 0.07 | 92.08 ± 0.30 | 99.24 / 85.56 / 95.81 / 80.71 / 99.08 | 7.30 | 9.86 | 1,231,272 |
| NSMT reproduction | 98.41 ± 0.03 | 91.58 ± 0.06 | 99.16 / 84.31 / 95.65 / 79.89 / 98.89 | 19.06 | 23.86 | 1,226,112 |
| Spikformer-style | 98.55 ± 0.02 | 92.22 ± 0.21 | 99.24 / 85.28 / 96.04 / 81.38 / 99.17 | 7.87 | 9.55 | 904,488 |
| NSMT paper Table 5 (as printed) | 98.45 | 91.91 | not printed | 19.05 | not printed | not printed |

| query pop LIF (default) vs | Δ macro-F1 %p (42 / 7 / 2026) | mean | verdict | energy A/B |
|---|---|---|---|---|
| NSMT reproduction | +0.55 / +0.43 / +0.79 | +0.59 | better | 0.47 |
| Spikformer-style | -0.23 / -0.24 / +0.32 | -0.05 | no clear difference | 1.16 |
| NSMT paper Table 5 (one printed value) | | +0.26 | not judged | synapses 0.44 |

lr on MIT-BIH (80 / 20 split of the training file, validation picks lr 0.02): test 91.66 vs 91.50 at lr 0.01 in the same split; lr 0.02 was not run on the full training file.

Not run for this model: ETTh1, ETTm1, the other forecasting datasets, anomaly detection, UEA classification, and the paper's stopping rule (patience 2, last weights). CinC is not in the paper; results/flif-v1-followup-summary.md has this model (90 epochs) at 67.73 vs Spikformer-style (30 epochs) 65.74 over 5 folds.
