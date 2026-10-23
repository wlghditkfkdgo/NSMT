# NSMT T_h = 4 with neocortex streams aligned to the hippocampus steps `flif-v1-rhythm-t4k-cinc` (CINC2017) — summary

Best-validation weights on the test set; CinC score and macro-F1; mean ± sd over seeds (42, 7, 2026) (%).

| condition | n | score | macro-F1 | best val | val − test | total µJ (FFT) | neocortex firing | params |
|---|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 3 | 65.60 ± 1.48 | 61.94 ± 0.41 | 68.79 ± 2.19 | 3.19 ± 0.76 | 127.4 | n/a | 918088 |
| NSMT T = N, spikeDE LIF | 3 | 61.76 ± 1.40 | 59.01 ± 0.95 | 66.00 ± 1.03 | 4.23 ± 0.57 | 3419.3 | 0.0792 ± 0.0005 | 1262592 |
| NSMT T = N, f-LIF | 3 | 61.50 ± 1.20 | 58.59 ± 0.90 | 65.68 ± 0.12 | 4.18 ± 1.11 | 3480.0 | 0.0881 ± 0.0001 | 1262592 |
| NSMT T = N, population f-LIF | 3 | 60.94 ± 2.24 | 55.05 ± 1.76 | 65.88 ± 0.60 | 4.94 ± 1.64 | 3247.4 | 0.0398 ± 0.0022 | 1267200 |
| NSMT T_h 4, spikeDE LIF (copied to the 4 steps) | 3 | 63.42 ± 0.85 | 55.80 ± 0.88 | 68.37 ± 0.97 | 4.95 ± 0.30 | 148.1 | 0.0590 ± 0.0017 | 1264456 |
| NSMT T_h 4, f-LIF (copied) | 3 | 61.84 ± 2.42 | 55.24 ± 1.07 | 67.18 ± 1.20 | 5.34 ± 1.24 | 148.2 | 0.0619 ± 0.0017 | 1264456 |
| NSMT T_h 4, population LIF with soma (copied) | 3 | 62.50 ± 0.84 | 57.82 ± 1.13 | 67.85 ± 1.76 | 5.36 ± 1.07 | 142.7 | 0.0515 ± 0.0062 | 1269064 |
| NSMT T_h 4, population f-LIF with soma (copied) | 3 | 58.66 ± 2.25 | 52.12 ± 1.11 | 64.12 ± 1.83 | 5.46 ± 0.43 | 160.0 | 0.0314 ± 0.0004 | 1269064 |
| NSMT T_h 4, 4 LIF streams = 4 steps | 3 | 63.52 ± 0.38 | 58.99 ± 0.88 | 68.03 ± 0.81 | 4.51 ± 0.44 | 149.1 | 0.0679 ± 0.0012 | 1271368 |
| NSMT T_h 4, 4 f-LIF streams = 4 steps | 3 | 58.33 ± 0.96 | 53.00 ± 1.42 | 64.59 ± 1.30 | 6.26 ± 1.97 | 162.3 | 0.0431 ± 0.0015 | 1271368 |

## Pre-registered comparisons (%p, paired by seed; better = mean >= +0.5 and A > B in >= 2 of 3 seeds)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| K1 streams = steps vs population with soma (copied) | NSMT T_h 4, 4 LIF streams = 4 steps | NSMT T_h 4, population LIF with soma (copied) | +1.39 / +1.38 / +0.31 | +1.02 | better | 1.05 |  |
| K1 streams = steps vs population with soma (copied) | NSMT T_h 4, 4 f-LIF streams = 4 steps | NSMT T_h 4, population f-LIF with soma (copied) | +2.26 / -0.24 / -3.03 | -0.33 | no clear difference | 1.01 |  |
| K2 streams = steps vs one neuron (copied) | NSMT T_h 4, 4 LIF streams = 4 steps | NSMT T_h 4, spikeDE LIF (copied to the 4 steps) | +0.16 / -0.61 / +0.75 | +0.10 | no clear difference | 1.01 |  |
| K2 streams = steps vs one neuron (copied) | NSMT T_h 4, 4 f-LIF streams = 4 steps | NSMT T_h 4, f-LIF (copied) | -2.04 / -2.26 / -6.24 | -3.52 | worse | 1.10 |  |
| K3 vs Spikformer (criterion 2) | NSMT T_h 4, 4 LIF streams = 4 steps | Spikformer-style SSA (T=4) | -4.17 / -0.62 / -1.44 | -2.08 | worse | 1.17 | not met |
| K3 vs Spikformer (criterion 2) | NSMT T_h 4, 4 f-LIF streams = 4 steps | Spikformer-style SSA (T=4) | -7.92 / -6.98 / -6.90 | -7.27 | worse | 1.27 | not met |
| K4 (report) new baselines vs T = N | NSMT T_h 4, f-LIF (copied) | NSMT T = N, f-LIF | +1.25 / -2.36 / +2.12 | +0.34 | no clear difference | 0.04 |  |
| K4 (report) population LIF vs LIF, copied | NSMT T_h 4, population LIF with soma (copied) | NSMT T_h 4, spikeDE LIF (copied to the 4 steps) | -1.22 / -1.98 / +0.44 | -0.92 | worse | 0.96 |  |
