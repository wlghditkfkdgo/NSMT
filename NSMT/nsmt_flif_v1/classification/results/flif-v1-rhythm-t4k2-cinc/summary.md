# Stream-NSMT controls `flif-v1-rhythm-t4k2-cinc` (CINC2017) — summary

Best-validation weights on the test set; CinC fold 0; mean ± sd over seeds (42, 7, 2026) (%).

| condition | n | score | macro-F1 | best val | val − test | total µJ (FFT) | neocortex firing | params |
|---|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 3 | 65.60 ± 1.48 | 61.94 ± 0.41 | 68.79 ± 2.19 | 3.19 ± 0.76 | 127.4 | n/a | 918088 |
| T_h 4, LIF copied | 3 | 63.42 ± 0.85 | 55.80 ± 0.88 | 68.37 ± 0.97 | 4.95 ± 0.30 | 148.1 | 0.0590 ± 0.0017 | 1264456 |
| T_h 4, f-LIF copied | 3 | 61.84 ± 2.42 | 55.24 ± 1.07 | 67.18 ± 1.20 | 5.34 ± 1.24 | 148.2 | 0.0619 ± 0.0017 | 1264456 |
| T_h 4, LIF streams = steps (tau 4 -> 32) | 3 | 63.52 ± 0.38 | 58.99 ± 0.88 | 68.03 ± 0.81 | 4.51 ± 0.44 | 149.1 | 0.0679 ± 0.0012 | 1271368 |
| T_h 4, f-LIF streams = steps (tau 4 -> 32) | 3 | 58.33 ± 0.96 | 53.00 ± 1.42 | 64.59 ± 1.30 | 6.26 ± 1.97 | 162.3 | 0.0431 ± 0.0015 | 1271368 |
| T_h 4, LIF streams, reversed (tau 32 -> 4) | 3 | 62.55 ± 1.60 | 56.48 ± 1.63 | 68.69 ± 1.05 | 6.14 ± 2.01 | 150.0 | 0.0711 ± 0.0015 | 1271368 |
| T_h 4, f-LIF streams, reversed | 3 | 58.85 ± 2.28 | 53.89 ± 1.57 | 64.05 ± 0.61 | 5.20 ± 2.87 | 160.8 | 0.0453 ± 0.0027 | 1271368 |
| T_h 4, LIF streams both directions fused (charges + difference) | 3 | 62.80 ± 0.70 | 57.78 ± 1.21 | 68.13 ± 0.97 | 5.34 ± 0.27 | 1136.7 | 0.0508 ± 0.0055 | 2045512 |
| T_h 4, f-LIF streams both directions fused | 3 | 58.22 ± 1.52 | 52.26 ± 1.87 | 63.15 ± 1.37 | 4.93 ± 1.13 | 1162.3 | 0.0369 ± 0.0005 | 2045512 |

## Pre-registered comparisons (%p, paired by seed; better = mean >= +0.5 and A > B in >= 2 of 3 seeds)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| S1 order reversed | T_h 4, LIF streams, reversed (tau 32 -> 4) | T_h 4, LIF streams = steps (tau 4 -> 32) | +0.97 / -2.93 / -0.95 | -0.97 | worse | 1.01 |  |
| S1 order reversed | T_h 4, f-LIF streams, reversed | T_h 4, f-LIF streams = steps (tau 4 -> 32) | -1.39 / -0.27 / +3.23 | +0.52 | no clear difference | 0.99 |  |
| S2 both directions fused | T_h 4, LIF streams both directions fused (charges + difference) | T_h 4, LIF streams = steps (tau 4 -> 32) | -0.02 / -0.46 / -1.69 | -0.72 | worse | 7.62 |  |
| S2 both directions fused | T_h 4, f-LIF streams both directions fused | T_h 4, f-LIF streams = steps (tau 4 -> 32) | -2.31 / +2.50 / -0.52 | -0.11 | no clear difference | 7.16 |  |
| S3 vs one neuron copied | T_h 4, LIF streams, reversed (tau 32 -> 4) | T_h 4, LIF copied | +1.13 / -3.53 / -0.20 | -0.87 | worse | 1.01 |  |
| S3 vs one neuron copied | T_h 4, LIF streams both directions fused (charges + difference) | T_h 4, LIF copied | +0.14 / -1.07 / -0.93 | -0.62 | worse | 7.68 |  |
| S3 vs one neuron copied | T_h 4, f-LIF streams, reversed | T_h 4, f-LIF copied | -3.43 / -2.53 / -3.01 | -2.99 | worse | 1.08 |  |
| S3 vs one neuron copied | T_h 4, f-LIF streams both directions fused | T_h 4, f-LIF copied | -4.36 / +0.24 / -6.76 | -3.63 | worse | 7.84 |  |
| S4 vs Spikformer (criterion 2) | T_h 4, LIF streams, reversed (tau 32 -> 4) | Spikformer-style SSA (T=4) | -3.20 / -3.55 / -2.38 | -3.05 | worse | 1.18 | not met |
| S4 vs Spikformer (criterion 2) | T_h 4, LIF streams both directions fused (charges + difference) | Spikformer-style SSA (T=4) | -4.19 / -1.09 / -3.12 | -2.80 | worse | 8.92 | not met |
| S4 vs Spikformer (criterion 2) | T_h 4, f-LIF streams, reversed | Spikformer-style SSA (T=4) | -9.30 / -7.26 / -3.67 | -6.75 | worse | 1.26 | not met |
| S4 vs Spikformer (criterion 2) | T_h 4, f-LIF streams both directions fused | Spikformer-style SSA (T=4) | -10.23 / -4.48 / -7.43 | -7.38 | worse | 9.13 | not met |
