# NSMT aux-loss options `flif-v1-rhythm-aux-shd` (SHD) — summary

Best-validation weights on the test set; test accuracy; mean ± sd over seeds (42, 7, 2026, 123, 1024) (%). Baseline rows and spk_ssa come from `results/flif-v1-rhythm-shd/raw.txt`. Final-epoch train CE / MSE from each run's log/best_log_0.csv (MSE is logged even when the aux loss is off).

| neuron | aux setting | n | score | macro-F1 | last-epoch score | final train CE | final train MSE | total µJ (FFT) | neocortex firing | time s |
|---|---|---|---|---|---|---|---|---|---|---|
| NSMT, spikeDE LIF | aux off (alpha 0, plain CE) | 5 | 80.50 ± 1.25 | 80.35 ± 1.39 | 80.74 ± 0.87 | 0.0030 | 0.1154 | 26471.3 | 0.0795 | 6439 |
| NSMT, spikeDE LIF | alpha 0.1, DCT target (stage 9 baseline) | 5 | 80.37 ± 1.05 | 80.13 ± 1.10 | 80.76 ± 0.57 | 0.0029 | 0.0146 | 26484.6 | 0.0825 | 7516 |
| NSMT, spikeDE LIF | alpha 0.5, DCT target | 5 | 80.08 ± 1.40 | 80.03 ± 1.31 | 80.53 ± 0.41 | 0.0030 | 0.0125 | 26483.2 | 0.0874 | 6412 |
| NSMT, spikeDE LIF | alpha 0.9, DCT target | 5 | 80.24 ± 1.05 | 80.13 ± 1.04 | 80.13 ± 0.36 | 0.0034 | 0.0112 | 26470.9 | 0.0960 | 7759 |
| NSMT, spikeDE LIF | alpha 0.1, full-signal target | 5 | 79.78 ± 0.98 | 79.65 ± 1.09 | 80.89 ± 0.35 | 0.0029 | 0.0691 | 26484.1 | 0.0810 | 6402 |
| NSMT, population f-LIF (calibrated) | aux off (alpha 0, plain CE) | 0 | n/a | | | | | | | |
| NSMT, population f-LIF (calibrated) | alpha 0.1, DCT target (stage 9 baseline) | 5 | 86.32 ± 0.92 | 85.46 ± 1.04 | 86.34 ± 0.73 | 0.0021 | 0.0229 | 26508.1 | 0.1197 | 11047 |
| NSMT, population f-LIF (calibrated) | alpha 0.5, DCT target | 0 | n/a | | | | | | | |
| NSMT, population f-LIF (calibrated) | alpha 0.9, DCT target | 0 | n/a | | | | | | | |
| NSMT, population f-LIF (calibrated) | alpha 0.1, full-signal target | 0 | n/a | | | | | | | |
| NSMT, f-LIF | aux off (alpha 0, plain CE) | 0 | n/a | | | | | | | |
| NSMT, f-LIF | alpha 0.1, DCT target (stage 9 baseline) | 5 | 81.34 ± 0.61 | 81.16 ± 0.62 | 80.87 ± 0.38 | 0.0030 | 0.0153 | 26484.1 | 0.0761 | 11027 |
| NSMT, f-LIF | alpha 0.5, DCT target | 0 | n/a | | | | | | | |
| NSMT, f-LIF | alpha 0.9, DCT target | 0 | n/a | | | | | | | |
| NSMT, f-LIF | alpha 0.1, full-signal target | 0 | n/a | | | | | | | |
| Hippo-only, Spikformer SSA (T=4) | – | 5 | 86.75 ± 1.32 | 86.41 ± 1.35 | 86.77 ± 0.97 | | | 1071.2 | n/a | 902 |

## Pre-registered comparisons (%p, paired by seed)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| X1 aux loss off | NSMT, spikeDE LIF / aux off (alpha 0, plain CE) | NSMT, spikeDE LIF / alpha 0.1, DCT target (stage 9 baseline) | +0.09 / -1.19 / -0.22 / -0.35 / +2.34 | +0.13 | no clear difference | 1.00 |  |
| X2 full-signal target | NSMT, spikeDE LIF / alpha 0.1, full-signal target | NSMT, spikeDE LIF / alpha 0.1, DCT target (stage 9 baseline) | -1.10 / -0.22 / -1.77 / -0.66 / +0.80 | -0.59 | worse | 1.00 |  |
| X3 alpha 0.5 | NSMT, spikeDE LIF / alpha 0.5, DCT target | NSMT, spikeDE LIF / alpha 0.1, DCT target (stage 9 baseline) | -3.75 / -0.13 / +1.59 / +0.31 / +0.53 | -0.29 | no clear difference | 1.00 |  |
| X3 alpha 0.9 | NSMT, spikeDE LIF / alpha 0.9, DCT target | NSMT, spikeDE LIF / alpha 0.1, DCT target (stage 9 baseline) | -1.90 / -0.97 / +0.00 / +0.88 / +1.33 | -0.13 | no clear difference | 1.00 |  |
| X1 aux loss off | NSMT, population f-LIF (calibrated) / aux off (alpha 0, plain CE) | NSMT, population f-LIF (calibrated) / alpha 0.1, DCT target (stage 9 baseline) |  | +nan | incomplete | nan |  |
| X2 full-signal target | NSMT, population f-LIF (calibrated) / alpha 0.1, full-signal target | NSMT, population f-LIF (calibrated) / alpha 0.1, DCT target (stage 9 baseline) |  | +nan | incomplete | nan |  |
| X3 alpha 0.5 | NSMT, population f-LIF (calibrated) / alpha 0.5, DCT target | NSMT, population f-LIF (calibrated) / alpha 0.1, DCT target (stage 9 baseline) |  | +nan | incomplete | nan |  |
| X3 alpha 0.9 | NSMT, population f-LIF (calibrated) / alpha 0.9, DCT target | NSMT, population f-LIF (calibrated) / alpha 0.1, DCT target (stage 9 baseline) |  | +nan | incomplete | nan |  |
| X1 aux loss off | NSMT, f-LIF / aux off (alpha 0, plain CE) | NSMT, f-LIF / alpha 0.1, DCT target (stage 9 baseline) |  | +nan | incomplete | nan |  |
| X2 full-signal target | NSMT, f-LIF / alpha 0.1, full-signal target | NSMT, f-LIF / alpha 0.1, DCT target (stage 9 baseline) |  | +nan | incomplete | nan |  |
| X3 alpha 0.5 | NSMT, f-LIF / alpha 0.5, DCT target | NSMT, f-LIF / alpha 0.1, DCT target (stage 9 baseline) |  | +nan | incomplete | nan |  |
| X3 alpha 0.9 | NSMT, f-LIF / alpha 0.9, DCT target | NSMT, f-LIF / alpha 0.1, DCT target (stage 9 baseline) |  | +nan | incomplete | nan |  |
| X4 vs Hippo-only (Spikformer) | NSMT, spikeDE LIF / aux off (alpha 0, plain CE) | Hippo-only, Spikformer SSA (T=4) | -5.92 / -7.77 / -5.61 / -7.69 / -4.24 | -6.25 | worse | 24.71 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, spikeDE LIF / alpha 0.1, full-signal target | Hippo-only, Spikformer SSA (T=4) | -7.11 / -6.80 / -7.16 / -7.99 / -5.79 | -6.97 | worse | 24.72 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, spikeDE LIF / alpha 0.5, DCT target | Hippo-only, Spikformer SSA (T=4) | -9.76 / -6.71 / -3.80 / -7.02 / -6.05 | -6.67 | worse | 24.72 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, spikeDE LIF / alpha 0.9, DCT target | Hippo-only, Spikformer SSA (T=4) | -7.91 / -7.55 / -5.39 / -6.45 / -5.26 | -6.51 | worse | 24.71 | not met |
| X4 vs Hippo-only (Spikformer) | NSMT, population f-LIF (calibrated) / aux off (alpha 0, plain CE) | Hippo-only, Spikformer SSA (T=4) |  | +nan | incomplete | nan |  |
| X4 vs Hippo-only (Spikformer) | NSMT, population f-LIF (calibrated) / alpha 0.1, full-signal target | Hippo-only, Spikformer SSA (T=4) |  | +nan | incomplete | nan |  |
| X4 vs Hippo-only (Spikformer) | NSMT, population f-LIF (calibrated) / alpha 0.5, DCT target | Hippo-only, Spikformer SSA (T=4) |  | +nan | incomplete | nan |  |
| X4 vs Hippo-only (Spikformer) | NSMT, population f-LIF (calibrated) / alpha 0.9, DCT target | Hippo-only, Spikformer SSA (T=4) |  | +nan | incomplete | nan |  |
| X4 vs Hippo-only (Spikformer) | NSMT, f-LIF / aux off (alpha 0, plain CE) | Hippo-only, Spikformer SSA (T=4) |  | +nan | incomplete | nan |  |
| X4 vs Hippo-only (Spikformer) | NSMT, f-LIF / alpha 0.1, full-signal target | Hippo-only, Spikformer SSA (T=4) |  | +nan | incomplete | nan |  |
| X4 vs Hippo-only (Spikformer) | NSMT, f-LIF / alpha 0.5, DCT target | Hippo-only, Spikformer SSA (T=4) |  | +nan | incomplete | nan |  |
| X4 vs Hippo-only (Spikformer) | NSMT, f-LIF / alpha 0.9, DCT target | Hippo-only, Spikformer SSA (T=4) |  | +nan | incomplete | nan |  |
