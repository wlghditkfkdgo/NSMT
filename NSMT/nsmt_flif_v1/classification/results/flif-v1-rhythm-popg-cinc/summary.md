# Tau-group Neo-mixer with linear fusions `flif-v1-rhythm-popg-cinc` (CINC2017) — summary

Best-validation weights on the test set; CinC fold 0; mean ± sd over seeds (42, 7, 2026) (%).

| condition | n | score | macro-F1 | best val | val − test | total µJ (FFT) | neocortex firing | params |
|---|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 3 | 65.60 ± 1.48 | 61.94 ± 0.41 | 68.79 ± 2.19 | 3.19 ± 0.76 | 127.4 | n/a | 918088 |
| Neo-mixer OR, soma-less tau groups, LIF | 3 | 59.27 ± 0.45 | 55.01 ± 1.30 | 61.30 ± 0.83 | 2.02 ± 0.74 | 26.0 | 0.1023 ± 0.0021 | 916224 |
| linear fusion of spikes, soma-less tau groups, LIF | 3 | 57.19 ± 0.07 | 52.53 ± 1.02 | 59.78 ± 0.42 | 2.59 ± 0.50 | 29.6 | 0.0863 ± 0.0012 | 1211904 |
| linear fusion of charges, soma-less tau groups, LIF | 3 | 57.61 ± 1.52 | 53.59 ± 1.22 | 60.25 ± 1.91 | 2.65 ± 3.36 | 185.4 | 0.0715 ± 0.0021 | 1211904 |
| charges + |H_f - H_r|, soma-less tau groups, LIF | 3 | 58.62 ± 1.07 | 55.04 ± 1.91 | 59.58 ± 2.00 | 0.96 ± 1.01 | 264.8 | 0.0795 ± 0.0004 | 1359360 |
| spikes + F xor R, soma-less tau groups, LIF | 3 | 58.06 ± 0.20 | 54.08 ± 0.20 | 59.28 ± 2.03 | 1.22 ± 2.00 | 31.9 | 0.0834 ± 0.0014 | 1359360 |
| Neo-mixer OR, soma-less tau groups, f-LIF | 3 | 59.51 ± 1.81 | 54.34 ± 0.37 | 60.22 ± 1.40 | 0.70 ± 1.15 | 30.5 | 0.0872 ± 0.0034 | 916224 |
| linear fusion of spikes, soma-less tau groups, f-LIF | 3 | 57.41 ± 0.30 | 52.77 ± 1.11 | 60.38 ± 0.37 | 2.97 ± 0.60 | 34.0 | 0.0759 ± 0.0019 | 1211904 |
| linear fusion of charges, soma-less tau groups, f-LIF | 3 | 56.67 ± 1.24 | 51.07 ± 0.47 | 59.85 ± 1.12 | 3.18 ± 1.39 | 189.7 | 0.0655 ± 0.0013 | 1211904 |
| charges + |H_f - H_r|, soma-less tau groups, f-LIF | 3 | 57.31 ± 1.12 | 53.12 ± 0.56 | 59.77 ± 0.74 | 2.46 ± 1.12 | 269.2 | 0.0743 ± 0.0003 | 1359360 |
| spikes + F xor R, soma-less tau groups, f-LIF | 3 | 57.73 ± 0.37 | 53.47 ± 1.38 | 61.69 ± 1.06 | 3.96 ± 1.25 | 36.4 | 0.0745 ± 0.0008 | 1359360 |

## Pre-registered comparisons (%p, paired by seed; better = mean >= +0.5 and A > B in >= 2 of 3 seeds)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| G0 OR -> linear fusion | linear fusion of spikes, soma-less tau groups, LIF | Neo-mixer OR, soma-less tau groups, LIF | -2.54 / -1.72 / -1.99 | -2.08 | worse | 1.14 |  |
| G0 OR -> linear fusion | linear fusion of spikes, soma-less tau groups, f-LIF | Neo-mixer OR, soma-less tau groups, f-LIF | -0.06 / -2.34 / -3.92 | -2.11 | worse | 1.11 |  |
| G1 charges vs spikes (M1) | linear fusion of charges, soma-less tau groups, LIF | linear fusion of spikes, soma-less tau groups, LIF | -1.28 / +0.90 / +1.62 | +0.41 | no clear difference | 6.27 |  |
| G1 charges vs spikes (M1) | linear fusion of charges, soma-less tau groups, f-LIF | linear fusion of spikes, soma-less tau groups, f-LIF | -0.00 / -0.36 / -1.85 | -0.74 | worse | 5.58 |  |
| G2 + |H_f - H_r| (M2) | charges + |H_f - H_r|, soma-less tau groups, LIF | linear fusion of charges, soma-less tau groups, LIF | +1.89 / +1.82 / -0.67 | +1.01 | better | 1.43 |  |
| G2 + |H_f - H_r| (M2) | charges + |H_f - H_r|, soma-less tau groups, f-LIF | linear fusion of charges, soma-less tau groups, f-LIF | -0.84 / -0.59 / +3.36 | +0.64 | no clear difference | 1.42 |  |
| G3 charges + difference vs spikes (M3) | charges + |H_f - H_r|, soma-less tau groups, LIF | linear fusion of spikes, soma-less tau groups, LIF | +0.61 / +2.72 / +0.95 | +1.43 | better | 8.95 |  |
| G3 charges + difference vs spikes (M3) | charges + |H_f - H_r|, soma-less tau groups, f-LIF | linear fusion of spikes, soma-less tau groups, f-LIF | -0.85 / -0.95 / +1.51 | -0.10 | no clear difference | 7.92 |  |
| G4 + F xor R (X1) | spikes + F xor R, soma-less tau groups, LIF | linear fusion of spikes, soma-less tau groups, LIF | +0.65 / +0.97 / +0.97 | +0.86 | better | 1.08 |  |
| G4 + F xor R (X1) | spikes + F xor R, soma-less tau groups, f-LIF | linear fusion of spikes, soma-less tau groups, f-LIF | +0.35 / +0.38 / +0.25 | +0.33 | no clear difference | 1.07 |  |
| G5 soma-less vs soma (same fusion) | linear fusion of spikes, soma-less tau groups, LIF | mixlin, population lif with soma | -5.91 / -6.37 / -3.19 | -5.16 | worse | nan |  |
| G5 soma-less vs soma (same fusion) | linear fusion of charges, soma-less tau groups, LIF | mixlinmem, population lif with soma | -5.75 / -5.39 / -1.78 | -4.31 | worse | nan |  |
| G5 soma-less vs soma (same fusion) | charges + |H_f - H_r|, soma-less tau groups, LIF | mixlinmx, population lif with soma | -1.11 / -1.61 / -3.53 | -2.08 | worse | nan |  |
| G5 soma-less vs soma (same fusion) | spikes + F xor R, soma-less tau groups, LIF | mixlinxor, population lif with soma | -3.72 / -3.50 / -2.92 | -3.38 | worse | nan |  |
| G5 soma-less vs soma (same fusion) | linear fusion of spikes, soma-less tau groups, f-LIF | mixlin, population flif with soma | -3.57 / -4.83 / -5.76 | -4.72 | worse | nan |  |
| G5 soma-less vs soma (same fusion) | linear fusion of charges, soma-less tau groups, f-LIF | mixlinmem, population flif with soma | -3.05 / -3.14 / -6.34 | -4.18 | worse | nan |  |
| G5 soma-less vs soma (same fusion) | charges + |H_f - H_r|, soma-less tau groups, f-LIF | mixlinmx, population flif with soma | -5.17 / -3.35 / -2.95 | -3.83 | worse | nan |  |
| G5 soma-less vs soma (same fusion) | spikes + F xor R, soma-less tau groups, f-LIF | mixlinxor, population flif with soma | -3.04 / -3.70 / -3.51 | -3.41 | worse | nan |  |
| G6 vs Spikformer (criterion 2) | linear fusion of spikes, soma-less tau groups, LIF | Spikformer-style SSA (T=4) | -10.06 / -7.30 / -7.86 | -8.40 | worse | 0.23 | not met |
| G6 vs Spikformer (criterion 2) | linear fusion of charges, soma-less tau groups, LIF | Spikformer-style SSA (T=4) | -11.33 / -6.40 / -6.24 | -7.99 | worse | 1.46 | not met |
| G6 vs Spikformer (criterion 2) | charges + |H_f - H_r|, soma-less tau groups, LIF | Spikformer-style SSA (T=4) | -9.44 / -4.57 / -6.91 | -6.98 | worse | 2.08 | not met |
| G6 vs Spikformer (criterion 2) | spikes + F xor R, soma-less tau groups, LIF | Spikformer-style SSA (T=4) | -9.41 / -6.32 / -6.89 | -7.54 | worse | 0.25 | not met |
| G6 vs Spikformer (criterion 2) | linear fusion of spikes, soma-less tau groups, f-LIF | Spikformer-style SSA (T=4) | -9.81 / -6.73 / -8.03 | -8.19 | worse | 0.27 | not met |
| G6 vs Spikformer (criterion 2) | linear fusion of charges, soma-less tau groups, f-LIF | Spikformer-style SSA (T=4) | -9.82 / -7.09 / -9.88 | -8.93 | worse | 1.49 | not met |
| G6 vs Spikformer (criterion 2) | charges + |H_f - H_r|, soma-less tau groups, f-LIF | Spikformer-style SSA (T=4) | -10.66 / -7.68 / -6.52 | -8.29 | worse | 2.11 | not met |
| G6 vs Spikformer (criterion 2) | spikes + F xor R, soma-less tau groups, f-LIF | Spikformer-style SSA (T=4) | -9.46 / -6.35 / -7.78 | -7.86 | worse | 0.29 | not met |
