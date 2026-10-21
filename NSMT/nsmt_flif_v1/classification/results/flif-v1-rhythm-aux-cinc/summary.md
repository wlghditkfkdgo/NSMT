# NSMT aux-loss options `flif-v1-rhythm-aux-cinc` (CINC2017) — summary

Best-validation weights on the test set; CinC score (mean F1 of N, AF, Other); mean ± sd over seeds (42, 7, 2026) (%). Baseline rows and spk_ssa come from `results/flif-v1-rhythm-cinc/raw.txt`. Final-epoch train CE / MSE from each run's log/best_log_0.csv (MSE is logged even when the aux loss is off).

| neuron | aux setting | n | score | macro-F1 | last-epoch score | final train CE | final train MSE | total µJ (FFT) | neocortex firing | time s |
|---|---|---|---|---|---|---|---|---|---|---|
| NSMT, spikeDE LIF | aux off (alpha 0, plain CE) | 3 | 61.47 ± 1.37 | 58.43 ± 0.82 | 60.96 ± 1.30 | 0.3865 | 0.7376 | 3482.5 | 0.0575 | 5449 |
| NSMT, spikeDE LIF | alpha 0.1, DCT target (stage 9 baseline) | 3 | 61.76 ± 1.40 | 59.01 ± 0.95 | 61.61 ± 0.72 | 0.3323 | 0.2464 | 3419.3 | 0.0792 | 5144 |
| NSMT, spikeDE LIF | alpha 0.5, DCT target | 3 | 61.03 ± 0.44 | 58.19 ± 1.19 | 60.72 ± 0.85 | 0.3196 | 0.1271 | 3414.7 | 0.1053 | 4300 |
| NSMT, spikeDE LIF | alpha 0.9, DCT target | 3 | 61.02 ± 1.23 | 58.48 ± 1.43 | 60.58 ± 0.99 | 0.3701 | 0.0813 | 3379.9 | 0.1284 | 4151 |
| NSMT, spikeDE LIF | alpha 0.1, full-signal target | 3 | 61.20 ± 1.13 | 58.89 ± 0.40 | 61.41 ± 0.73 | 0.3267 | 0.3500 | 3463.7 | 0.0810 | 4675 |
| NSMT, population f-LIF (calibrated) | aux off (alpha 0, plain CE) | 3 | 60.77 ± 0.94 | 56.62 ± 1.44 | 60.83 ± 1.71 | 0.4524 | 0.7449 | 3259.6 | 0.0667 | 6435 |
| NSMT, population f-LIF (calibrated) | alpha 0.1, DCT target (stage 9 baseline) | 3 | 60.94 ± 2.24 | 55.05 ± 1.76 | 60.57 ± 0.54 | 0.3730 | 0.5601 | 3247.4 | 0.0398 | 7135 |
| NSMT, population f-LIF (calibrated) | alpha 0.5, DCT target | 3 | 61.40 ± 0.60 | 56.47 ± 0.29 | 61.13 ± 1.13 | 0.3391 | 0.3634 | 3323.3 | 0.0595 | 5804 |
| NSMT, population f-LIF (calibrated) | alpha 0.9, DCT target | 3 | 61.67 ± 0.62 | 56.49 ± 0.74 | 60.71 ± 0.64 | 0.4617 | 0.2655 | 3320.8 | 0.1148 | 6180 |
| NSMT, population f-LIF (calibrated) | alpha 0.1, full-signal target | 3 | 60.31 ± 0.17 | 55.66 ± 1.09 | 59.67 ± 1.20 | 0.3641 | 0.7044 | 3242.0 | 0.0366 | 6017 |
| NSMT, f-LIF | aux off (alpha 0, plain CE) | 3 | 61.36 ± 1.06 | 58.80 ± 0.76 | 61.06 ± 0.65 | 0.3863 | 0.7356 | 3512.9 | 0.0544 | 7103 |
| NSMT, f-LIF | alpha 0.1, DCT target (stage 9 baseline) | 3 | 61.50 ± 1.20 | 58.59 ± 0.90 | 60.53 ± 0.61 | 0.3172 | 0.2093 | 3480.0 | 0.0881 | 7284 |
| NSMT, f-LIF | alpha 0.5, DCT target | 3 | 60.86 ± 0.21 | 58.36 ± 0.08 | 60.40 ± 0.56 | 0.3081 | 0.1285 | 3471.2 | 0.1112 | 8648 |
| NSMT, f-LIF | alpha 0.9, DCT target | 3 | 60.66 ± 0.53 | 57.04 ± 1.04 | 60.36 ± 0.92 | 0.3420 | 0.0894 | 3379.0 | 0.1285 | 8940 |
| NSMT, f-LIF | alpha 0.1, full-signal target | 3 | 61.38 ± 1.15 | 58.99 ± 1.82 | 60.87 ± 0.41 | 0.3112 | 0.3082 | 3452.4 | 0.0916 | 8015 |
| Hippo-only, Spikformer SSA (T=4) | – | 3 | 65.60 ± 1.48 | 61.94 ± 0.41 | 65.36 ± 1.17 | | | 127.4 | n/a | 400 |

## Pre-registered comparisons (%p, paired by seed)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| X1 aux loss off | NSMT, spikeDE LIF / aux off (alpha 0, plain CE) | NSMT, spikeDE LIF / alpha 0.1, DCT target (stage 9 baseline) | +1.04 / +0.07 / -1.98 | -0.29 | no clear difference | 1.02 |  |
| X2 full-signal target | NSMT, spikeDE LIF / alpha 0.1, full-signal target | NSMT, spikeDE LIF / alpha 0.1, DCT target (stage 9 baseline) | -1.48 / +2.23 / -2.44 | -0.56 | worse | 1.01 |  |
| X3 alpha 0.5 | NSMT, spikeDE LIF / alpha 0.5, DCT target | NSMT, spikeDE LIF / alpha 0.1, DCT target (stage 9 baseline) | -1.10 / +0.44 / -1.53 | -0.73 | worse | 1.00 |  |
| X3 alpha 0.9 | NSMT, spikeDE LIF / alpha 0.9, DCT target | NSMT, spikeDE LIF / alpha 0.1, DCT target (stage 9 baseline) | +0.01 / +1.18 / -3.43 | -0.75 | no clear difference | 0.99 |  |
| X1 aux loss off | NSMT, population f-LIF (calibrated) / aux off (alpha 0, plain CE) | NSMT, population f-LIF (calibrated) / alpha 0.1, DCT target (stage 9 baseline) | -2.33 / -1.39 / +3.21 | -0.17 | no clear difference | 1.00 |  |
| X2 full-signal target | NSMT, population f-LIF (calibrated) / alpha 0.1, full-signal target | NSMT, population f-LIF (calibrated) / alpha 0.1, DCT target (stage 9 baseline) | -2.85 / -0.96 / +1.94 | -0.62 | worse | 1.00 |  |
| X3 alpha 0.5 | NSMT, population f-LIF (calibrated) / alpha 0.5, DCT target | NSMT, population f-LIF (calibrated) / alpha 0.1, DCT target (stage 9 baseline) | -2.29 / +0.60 / +3.06 | +0.46 | no clear difference | 1.02 |  |
| X3 alpha 0.9 | NSMT, population f-LIF (calibrated) / alpha 0.9, DCT target | NSMT, population f-LIF (calibrated) / alpha 0.1, DCT target (stage 9 baseline) | -0.68 / -0.18 / +3.05 | +0.73 | no clear difference | 1.02 |  |
| X1 aux loss off | NSMT, f-LIF / aux off (alpha 0, plain CE) | NSMT, f-LIF / alpha 0.1, DCT target (stage 9 baseline) | +2.25 / -0.60 / -2.07 | -0.14 | no clear difference | 1.01 |  |
| X2 full-signal target | NSMT, f-LIF / alpha 0.1, full-signal target | NSMT, f-LIF / alpha 0.1, DCT target (stage 9 baseline) | +1.38 / -1.87 / +0.14 | -0.12 | no clear difference | 0.99 |  |
| X3 alpha 0.5 | NSMT, f-LIF / alpha 0.5, DCT target | NSMT, f-LIF / alpha 0.1, DCT target (stage 9 baseline) | +0.80 / -1.42 / -1.30 | -0.64 | worse | 1.00 |  |
| X3 alpha 0.9 | NSMT, f-LIF / alpha 0.9, DCT target | NSMT, f-LIF / alpha 0.1, DCT target (stage 9 baseline) | +1.15 / -1.67 / -2.00 | -0.84 | worse | 0.97 |  |
| X4 vs Hippo-only (Spikformer) | NSMT, spikeDE LIF / aux off (alpha 0, plain CE) | Hippo-only, Spikformer SSA (T=4) | -4.26 / -4.07 / -4.04 | -4.12 | worse | 27.34 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, spikeDE LIF / alpha 0.1, full-signal target | Hippo-only, Spikformer SSA (T=4) | -6.78 / -1.91 / -4.50 | -4.40 | worse | 27.20 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, spikeDE LIF / alpha 0.5, DCT target | Hippo-only, Spikformer SSA (T=4) | -6.40 / -3.71 / -3.59 | -4.56 | worse | 26.81 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, spikeDE LIF / alpha 0.9, DCT target | Hippo-only, Spikformer SSA (T=4) | -5.29 / -2.96 / -5.49 | -4.58 | worse | 26.54 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, population f-LIF (calibrated) / aux off (alpha 0, plain CE) | Hippo-only, Spikformer SSA (T=4) | -6.58 / -4.54 / -3.37 | -4.83 | worse | 25.59 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, population f-LIF (calibrated) / alpha 0.1, full-signal target | Hippo-only, Spikformer SSA (T=4) | -7.11 / -4.11 / -4.63 | -5.28 | worse | 25.45 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, population f-LIF (calibrated) / alpha 0.5, DCT target | Hippo-only, Spikformer SSA (T=4) | -6.54 / -2.55 / -3.51 | -4.20 | worse | 26.09 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, population f-LIF (calibrated) / alpha 0.9, DCT target | Hippo-only, Spikformer SSA (T=4) | -4.93 / -3.33 / -3.52 | -3.93 | worse | 26.07 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, f-LIF / aux off (alpha 0, plain CE) | Hippo-only, Spikformer SSA (T=4) | -4.88 / -2.97 / -4.86 | -4.23 | worse | 27.58 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, f-LIF / alpha 0.1, full-signal target | Hippo-only, Spikformer SSA (T=4) | -5.75 / -4.24 / -2.65 | -4.21 | worse | 27.11 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, f-LIF / alpha 0.5, DCT target | Hippo-only, Spikformer SSA (T=4) | -6.33 / -3.78 / -4.08 | -4.73 | worse | 27.25 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, f-LIF / alpha 0.9, DCT target | Hippo-only, Spikformer SSA (T=4) | -5.97 / -4.03 / -4.79 | -4.93 | worse | 26.53 | not met |
