# Spikformer + T = N neocortex as extra keys / values `flif-v1-rhythm-addneo-cinc` (CINC2017) — summary

Best-validation weights on the test set; CinC score and macro-F1; mean ± sd over seeds (42, 7, 2026) (%). C2 is pre-registered as primary for LIF (same neuron); for the other neurons it also includes the neuron change. C3 is report only (the replace runs keep aux 0.1 and gradient share 0.5).

| condition | n | score | macro-F1 | best val | val − test | total µJ (FFT) | neocortex firing | params |
|---|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 3 | 65.60 ± 1.48 | 61.94 ± 0.41 | 68.79 ± 2.19 | 3.19 ± 0.76 | 127.4 | n/a | 918088 |
| Spikformer + neocortex added, LIF, memoryless (reset every token) | 3 | 65.84 ± 1.92 | 59.43 ± 0.81 | 69.21 ± 0.76 | 3.38 ± 2.55 | 161.6 | 0.0249 ± 0.0011 | 1560904 |
| Spikformer + T = N neocortex added, spikeDE LIF | 3 | 63.34 ± 1.55 | 57.32 ± 0.99 | 67.38 ± 1.18 | 4.04 ± 1.53 | 165.4 | 0.0355 ± 0.0014 | 1560904 |
| Spikformer + T = N neocortex added, f-LIF | 3 | 64.35 ± 0.43 | 59.89 ± 0.90 | 68.24 ± 0.20 | 3.89 ± 0.24 | 170.1 | 0.0338 ± 0.0018 | 1560904 |
| Spikformer + T = N neocortex added, population LIF (psi 0.5) | 3 | 66.06 ± 0.71 | 59.62 ± 2.28 | 70.21 ± 1.01 | 4.14 ± 0.40 | 172.8 | 0.0465 ± 0.0073 | 1565512 |
| Spikformer + T = N neocortex added, population f-LIF (psi 1) | 3 | 64.72 ± 0.40 | 59.15 ± 0.98 | 69.89 ± 1.13 | 5.17 ± 0.82 | 191.5 | 0.0359 ± 0.0029 | 1565512 |
| NSMT T_h 4, neocortex replaces the own keys / values, spikeDE LIF | 3 | 63.42 ± 0.85 | 55.80 ± 0.88 | 68.37 ± 0.97 | 4.95 ± 0.30 | 148.1 | 0.0590 ± 0.0017 | 1264456 |
| NSMT T_h 4, neocortex replaces the own keys / values, f-LIF | 3 | 61.84 ± 2.42 | 55.24 ± 1.07 | 67.18 ± 1.20 | 5.34 ± 1.24 | 148.2 | 0.0619 ± 0.0017 | 1264456 |
| NSMT T_h 4, neocortex replaces the own keys / values, population LIF (psi 0.5) | 3 | 62.50 ± 0.84 | 57.82 ± 1.13 | 67.85 ± 1.76 | 5.36 ± 1.07 | 142.7 | 0.0515 ± 0.0062 | 1269064 |
| NSMT T_h 4, neocortex replaces the own keys / values, population f-LIF (psi 1) | 3 | 58.66 ± 2.25 | 52.12 ± 1.11 | 64.12 ± 1.83 | 5.46 ± 0.43 | 160.0 | 0.0314 ± 0.0004 | 1269064 |

## Pre-registered comparisons (%p, paired by seed; better = mean >= +0.5 and A > B in >= 2 of 3 seeds)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| C1 vs Spikformer (criterion 2) | Spikformer + T = N neocortex added, spikeDE LIF | Spikformer-style SSA (T=4) | -3.75 / +0.38 / -3.42 | -2.26 | worse | 1.30 | not met |
| C1 vs Spikformer (criterion 2) | Spikformer + T = N neocortex added, f-LIF | Spikformer-style SSA (T=4) | -2.58 / -0.55 / -0.60 | -1.24 | worse | 1.34 | not met |
| C1 vs Spikformer (criterion 2) | Spikformer + T = N neocortex added, population LIF (psi 0.5) | Spikformer-style SSA (T=4) | -0.44 / +1.55 / +0.29 | +0.47 | no clear difference | 1.36 | not met |
| C1 vs Spikformer (criterion 2) | Spikformer + T = N neocortex added, population f-LIF (psi 1) | Spikformer-style SSA (T=4) | -2.40 / +0.63 / -0.85 | -0.87 | worse | 1.50 | not met |
| C1 vs Spikformer (criterion 2) | Spikformer + neocortex added, LIF, memoryless (reset every token) | Spikformer-style SSA (T=4) | +0.60 / +1.20 / -1.08 | +0.24 | no clear difference | 1.27 | not met |
| C2 T = N memory vs memoryless neocortex | Spikformer + T = N neocortex added, spikeDE LIF | Spikformer + neocortex added, LIF, memoryless (reset every token) | -4.35 / -0.82 / -2.34 | -2.50 | worse | 1.02 |  |
| C2 T = N memory vs memoryless neocortex | Spikformer + T = N neocortex added, f-LIF | Spikformer + neocortex added, LIF, memoryless (reset every token) | -3.18 / -1.75 / +0.48 | -1.48 | worse | 1.05 |  |
| C2 T = N memory vs memoryless neocortex | Spikformer + T = N neocortex added, population LIF (psi 0.5) | Spikformer + neocortex added, LIF, memoryless (reset every token) | -1.05 / +0.35 / +1.37 | +0.23 | no clear difference | 1.07 |  |
| C2 T = N memory vs memoryless neocortex | Spikformer + T = N neocortex added, population f-LIF (psi 1) | Spikformer + neocortex added, LIF, memoryless (reset every token) | -3.00 / -0.57 / +0.23 | -1.11 | worse | 1.19 |  |
| C3 (report) add vs replace | Spikformer + T = N neocortex added, spikeDE LIF | NSMT T_h 4, neocortex replaces the own keys / values, spikeDE LIF | +0.58 / +0.40 / -1.23 | -0.08 | no clear difference | 1.12 |  |
| C3 (report) add vs replace | Spikformer + T = N neocortex added, f-LIF | NSMT T_h 4, neocortex replaces the own keys / values, f-LIF | +3.30 / +4.17 / +0.06 | +2.51 | better | 1.15 |  |
| C3 (report) add vs replace | Spikformer + T = N neocortex added, population LIF (psi 0.5) | NSMT T_h 4, neocortex replaces the own keys / values, population LIF (psi 0.5) | +5.11 / +3.55 / +2.04 | +3.57 | better | 1.21 |  |
| C3 (report) add vs replace | Spikformer + T = N neocortex added, population f-LIF (psi 1) | NSMT T_h 4, neocortex replaces the own keys / values, population f-LIF (psi 1) | +7.78 / +7.37 / +3.03 | +6.06 | better | 1.20 |  |
