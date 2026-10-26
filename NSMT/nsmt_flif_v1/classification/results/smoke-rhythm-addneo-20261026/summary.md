# Spikformer + T = N neocortex as extra keys / values `smoke-rhythm-addneo-20261026` (CINC2017) — summary

Best-validation weights on the test set; CinC score and macro-F1; mean ± sd over seeds (42, 7, 2026) (%). C2 is pre-registered as primary for LIF (same neuron); for the other neurons it also includes the neuron change. C3 is report only (the replace runs keep aux 0.1 and gradient share 0.5).

| condition | n | score | macro-F1 | best val | val − test | total µJ (FFT) | neocortex firing | params |
|---|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 3 | 65.60 ± 1.48 | 61.94 ± 0.41 | 68.79 ± 2.19 | 3.19 ± 0.76 | 127.4 | n/a | 918088 |
| Spikformer + neocortex added, LIF, memoryless (reset every token) | 1 | 33.34 ± 0.00 | 25.01 ± 0.00 | 34.08 ± 0.00 | 0.74 ± 0.00 | 145.3 | 0.0306 ± 0.0000 | 1560904 |
| Spikformer + T = N neocortex added, spikeDE LIF | 1 | 37.83 ± 0.00 | 28.38 ± 0.00 | 37.95 ± 0.00 | 0.12 ± 0.00 | 142.2 | 0.0400 ± 0.0000 | 1560904 |
| Spikformer + T = N neocortex added, f-LIF | 1 | 36.25 ± 0.00 | 27.19 ± 0.00 | 36.81 ± 0.00 | 0.56 ± 0.00 | 144.6 | 0.0358 ± 0.0000 | 1560904 |
| Spikformer + T = N neocortex added, population LIF (psi 0.5) | 1 | 33.15 ± 0.00 | 24.86 ± 0.00 | 34.46 ± 0.00 | 1.31 ± 0.00 | 150.4 | 0.0408 ± 0.0000 | 1565512 |
| Spikformer + T = N neocortex added, population f-LIF (psi 1) | 1 | 32.96 ± 0.00 | 24.72 ± 0.00 | 33.02 ± 0.00 | 0.06 ± 0.00 | 176.1 | 0.0360 ± 0.0000 | 1565512 |
| NSMT T_h 4, neocortex replaces the own keys / values, spikeDE LIF | 3 | 63.42 ± 0.85 | 55.80 ± 0.88 | 68.37 ± 0.97 | 4.95 ± 0.30 | 148.1 | 0.0590 ± 0.0017 | 1264456 |
| NSMT T_h 4, neocortex replaces the own keys / values, f-LIF | 3 | 61.84 ± 2.42 | 55.24 ± 1.07 | 67.18 ± 1.20 | 5.34 ± 1.24 | 148.2 | 0.0619 ± 0.0017 | 1264456 |
| NSMT T_h 4, neocortex replaces the own keys / values, population LIF (psi 0.5) | 3 | 62.50 ± 0.84 | 57.82 ± 1.13 | 67.85 ± 1.76 | 5.36 ± 1.07 | 142.7 | 0.0515 ± 0.0062 | 1269064 |
| NSMT T_h 4, neocortex replaces the own keys / values, population f-LIF (psi 1) | 3 | 58.66 ± 2.25 | 52.12 ± 1.11 | 64.12 ± 1.83 | 5.46 ± 0.43 | 160.0 | 0.0314 ± 0.0004 | 1269064 |

## Pre-registered comparisons (%p, paired by seed; better = mean >= +0.5 and A > B in >= 2 of 3 seeds)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| C1 vs Spikformer (criterion 2) | Spikformer + T = N neocortex added, spikeDE LIF | Spikformer-style SSA (T=4) |  | +nan | incomplete | 1.12 |  |
| C1 vs Spikformer (criterion 2) | Spikformer + T = N neocortex added, f-LIF | Spikformer-style SSA (T=4) |  | +nan | incomplete | 1.14 |  |
| C1 vs Spikformer (criterion 2) | Spikformer + T = N neocortex added, population LIF (psi 0.5) | Spikformer-style SSA (T=4) |  | +nan | incomplete | 1.18 |  |
| C1 vs Spikformer (criterion 2) | Spikformer + T = N neocortex added, population f-LIF (psi 1) | Spikformer-style SSA (T=4) |  | +nan | incomplete | 1.38 |  |
| C1 vs Spikformer (criterion 2) | Spikformer + neocortex added, LIF, memoryless (reset every token) | Spikformer-style SSA (T=4) |  | +nan | incomplete | 1.14 |  |
| C2 T = N memory vs memoryless neocortex | Spikformer + T = N neocortex added, spikeDE LIF | Spikformer + neocortex added, LIF, memoryless (reset every token) |  | +nan | incomplete | 0.98 |  |
| C2 T = N memory vs memoryless neocortex | Spikformer + T = N neocortex added, f-LIF | Spikformer + neocortex added, LIF, memoryless (reset every token) |  | +nan | incomplete | 1.00 |  |
| C2 T = N memory vs memoryless neocortex | Spikformer + T = N neocortex added, population LIF (psi 0.5) | Spikformer + neocortex added, LIF, memoryless (reset every token) |  | +nan | incomplete | 1.03 |  |
| C2 T = N memory vs memoryless neocortex | Spikformer + T = N neocortex added, population f-LIF (psi 1) | Spikformer + neocortex added, LIF, memoryless (reset every token) |  | +nan | incomplete | 1.21 |  |
| C3 (report) add vs replace | Spikformer + T = N neocortex added, spikeDE LIF | NSMT T_h 4, neocortex replaces the own keys / values, spikeDE LIF |  | +nan | incomplete | 0.96 |  |
| C3 (report) add vs replace | Spikformer + T = N neocortex added, f-LIF | NSMT T_h 4, neocortex replaces the own keys / values, f-LIF |  | +nan | incomplete | 0.98 |  |
| C3 (report) add vs replace | Spikformer + T = N neocortex added, population LIF (psi 0.5) | NSMT T_h 4, neocortex replaces the own keys / values, population LIF (psi 0.5) |  | +nan | incomplete | 1.05 |  |
| C3 (report) add vs replace | Spikformer + T = N neocortex added, population f-LIF (psi 1) | NSMT T_h 4, neocortex replaces the own keys / values, population f-LIF (psi 1) |  | +nan | incomplete | 1.10 |  |
