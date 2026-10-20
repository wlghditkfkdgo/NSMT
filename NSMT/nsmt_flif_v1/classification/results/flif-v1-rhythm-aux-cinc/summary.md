# NSMT aux-loss options `flif-v1-rhythm-aux-cinc` (CINC2017) — summary

Best-validation weights on the test set; CinC score (mean F1 of N, AF, Other); mean ± sd over seeds (42, 7, 2026) (%). Baseline rows and spk_ssa come from `results/flif-v1-rhythm-cinc/raw.txt`. Final-epoch train CE / MSE from each run's log/best_log_0.csv (MSE is logged even when the aux loss is off).

| neuron | aux setting | n | score | macro-F1 | last-epoch score | final train CE | final train MSE | total µJ (FFT) | neocortex firing | time s |
|---|---|---|---|---|---|---|---|---|---|---|
| NSMT, spikeDE LIF | aux off (alpha 0, plain CE) | 3 | 61.47 ± 1.37 | 58.43 ± 0.82 | 60.96 ± 1.30 | 0.3865 | 0.7376 | 3482.5 | 0.0575 | 5449 |
| NSMT, spikeDE LIF | alpha 0.1, DCT target (stage 9 baseline) | 3 | 61.76 ± 1.40 | 59.01 ± 0.95 | 61.61 ± 0.72 | 0.3323 | 0.2464 | 3419.3 | 0.0792 | 5144 |
| NSMT, spikeDE LIF | alpha 0.5, DCT target | 3 | 61.03 ± 0.44 | 58.19 ± 1.19 | 60.72 ± 0.85 | 0.3196 | 0.1271 | 3414.7 | 0.1053 | 4300 |
| NSMT, spikeDE LIF | alpha 0.9, DCT target | 3 | 61.02 ± 1.23 | 58.48 ± 1.43 | 60.58 ± 0.99 | 0.3701 | 0.0813 | 3379.9 | 0.1284 | 4151 |
| NSMT, spikeDE LIF | alpha 0.1, full-signal target | 3 | 61.20 ± 1.13 | 58.89 ± 0.40 | 61.41 ± 0.73 | 0.3267 | 0.3500 | 3463.7 | 0.0810 | 4675 |
| NSMT, population f-LIF (calibrated) | aux off (alpha 0, plain CE) | 0 | n/a | | | | | | | |
| NSMT, population f-LIF (calibrated) | alpha 0.1, DCT target (stage 9 baseline) | 3 | 60.94 ± 2.24 | 55.05 ± 1.76 | 60.57 ± 0.54 | 0.3730 | 0.5601 | 3247.4 | 0.0398 | 7135 |
| NSMT, population f-LIF (calibrated) | alpha 0.5, DCT target | 0 | n/a | | | | | | | |
| NSMT, population f-LIF (calibrated) | alpha 0.9, DCT target | 0 | n/a | | | | | | | |
| NSMT, population f-LIF (calibrated) | alpha 0.1, full-signal target | 0 | n/a | | | | | | | |
| NSMT, f-LIF | aux off (alpha 0, plain CE) | 0 | n/a | | | | | | | |
| NSMT, f-LIF | alpha 0.1, DCT target (stage 9 baseline) | 3 | 61.50 ± 1.20 | 58.59 ± 0.90 | 60.53 ± 0.61 | 0.3172 | 0.2093 | 3480.0 | 0.0881 | 7284 |
| NSMT, f-LIF | alpha 0.5, DCT target | 0 | n/a | | | | | | | |
| NSMT, f-LIF | alpha 0.9, DCT target | 0 | n/a | | | | | | | |
| NSMT, f-LIF | alpha 0.1, full-signal target | 0 | n/a | | | | | | | |
| Hippo-only, Spikformer SSA (T=4) | – | 3 | 65.60 ± 1.48 | 61.94 ± 0.41 | 65.36 ± 1.17 | | | 127.4 | n/a | 400 |

## Pre-registered comparisons (%p, paired by seed)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| X1 aux loss off | NSMT, spikeDE LIF / aux off (alpha 0, plain CE) | NSMT, spikeDE LIF / alpha 0.1, DCT target (stage 9 baseline) | +1.04 / +0.07 / -1.98 | -0.29 | no clear difference | 1.02 |  |
| X2 full-signal target | NSMT, spikeDE LIF / alpha 0.1, full-signal target | NSMT, spikeDE LIF / alpha 0.1, DCT target (stage 9 baseline) | -1.48 / +2.23 / -2.44 | -0.56 | worse | 1.01 |  |
| X3 alpha 0.5 | NSMT, spikeDE LIF / alpha 0.5, DCT target | NSMT, spikeDE LIF / alpha 0.1, DCT target (stage 9 baseline) | -1.10 / +0.44 / -1.53 | -0.73 | worse | 1.00 |  |
| X3 alpha 0.9 | NSMT, spikeDE LIF / alpha 0.9, DCT target | NSMT, spikeDE LIF / alpha 0.1, DCT target (stage 9 baseline) | +0.01 / +1.18 / -3.43 | -0.75 | no clear difference | 0.99 |  |
| X1 aux loss off | NSMT, population f-LIF (calibrated) / aux off (alpha 0, plain CE) | NSMT, population f-LIF (calibrated) / alpha 0.1, DCT target (stage 9 baseline) |  | +nan | incomplete | nan |  |
| X2 full-signal target | NSMT, population f-LIF (calibrated) / alpha 0.1, full-signal target | NSMT, population f-LIF (calibrated) / alpha 0.1, DCT target (stage 9 baseline) |  | +nan | incomplete | nan |  |
| X3 alpha 0.5 | NSMT, population f-LIF (calibrated) / alpha 0.5, DCT target | NSMT, population f-LIF (calibrated) / alpha 0.1, DCT target (stage 9 baseline) |  | +nan | incomplete | nan |  |
| X3 alpha 0.9 | NSMT, population f-LIF (calibrated) / alpha 0.9, DCT target | NSMT, population f-LIF (calibrated) / alpha 0.1, DCT target (stage 9 baseline) |  | +nan | incomplete | nan |  |
| X1 aux loss off | NSMT, f-LIF / aux off (alpha 0, plain CE) | NSMT, f-LIF / alpha 0.1, DCT target (stage 9 baseline) |  | +nan | incomplete | nan |  |
| X2 full-signal target | NSMT, f-LIF / alpha 0.1, full-signal target | NSMT, f-LIF / alpha 0.1, DCT target (stage 9 baseline) |  | +nan | incomplete | nan |  |
| X3 alpha 0.5 | NSMT, f-LIF / alpha 0.5, DCT target | NSMT, f-LIF / alpha 0.1, DCT target (stage 9 baseline) |  | +nan | incomplete | nan |  |
| X3 alpha 0.9 | NSMT, f-LIF / alpha 0.9, DCT target | NSMT, f-LIF / alpha 0.1, DCT target (stage 9 baseline) |  | +nan | incomplete | nan |  |
| X4 vs Hippo-only (Spikformer) | NSMT, spikeDE LIF / aux off (alpha 0, plain CE) | Hippo-only, Spikformer SSA (T=4) | -4.26 / -4.07 / -4.04 | -4.12 | worse | 27.34 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, spikeDE LIF / alpha 0.1, full-signal target | Hippo-only, Spikformer SSA (T=4) | -6.78 / -1.91 / -4.50 | -4.40 | worse | 27.20 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, spikeDE LIF / alpha 0.5, DCT target | Hippo-only, Spikformer SSA (T=4) | -6.40 / -3.71 / -3.59 | -4.56 | worse | 26.81 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, spikeDE LIF / alpha 0.9, DCT target | Hippo-only, Spikformer SSA (T=4) | -5.29 / -2.96 / -5.49 | -4.58 | worse | 26.54 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, population f-LIF (calibrated) / aux off (alpha 0, plain CE) | Hippo-only, Spikformer SSA (T=4) |  | +nan | incomplete | nan |  |
| X4 vs Hippo-only (Spikformer) | NSMT, population f-LIF (calibrated) / alpha 0.1, full-signal target | Hippo-only, Spikformer SSA (T=4) |  | +nan | incomplete | nan |  |
| X4 vs Hippo-only (Spikformer) | NSMT, population f-LIF (calibrated) / alpha 0.5, DCT target | Hippo-only, Spikformer SSA (T=4) |  | +nan | incomplete | nan |  |
| X4 vs Hippo-only (Spikformer) | NSMT, population f-LIF (calibrated) / alpha 0.9, DCT target | Hippo-only, Spikformer SSA (T=4) |  | +nan | incomplete | nan |  |
| X4 vs Hippo-only (Spikformer) | NSMT, f-LIF / aux off (alpha 0, plain CE) | Hippo-only, Spikformer SSA (T=4) |  | +nan | incomplete | nan |  |
| X4 vs Hippo-only (Spikformer) | NSMT, f-LIF / alpha 0.1, full-signal target | Hippo-only, Spikformer SSA (T=4) |  | +nan | incomplete | nan |  |
| X4 vs Hippo-only (Spikformer) | NSMT, f-LIF / alpha 0.5, DCT target | Hippo-only, Spikformer SSA (T=4) |  | +nan | incomplete | nan |  |
| X4 vs Hippo-only (Spikformer) | NSMT, f-LIF / alpha 0.9, DCT target | Hippo-only, Spikformer SSA (T=4) |  | +nan | incomplete | nan |  |
