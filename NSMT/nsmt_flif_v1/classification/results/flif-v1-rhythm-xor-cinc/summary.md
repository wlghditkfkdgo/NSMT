# Spike-xor fusion Neo-mixer `flif-v1-rhythm-xor-cinc` (CINC2017) — summary

Best-validation weights on the test set; CinC fold 0; mean ± sd over seeds (42, 7, 2026) (%).

| condition | n | score | macro-F1 | best val | val − test | total µJ (FFT) | neocortex firing | params |
|---|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 3 | 65.60 ± 1.48 | 61.94 ± 0.41 | 68.79 ± 2.19 | 3.19 ± 0.76 | 127.4 | n/a | 918088 |
| Neo-mixer OR, spikeDE LIF | 3 | 60.54 ± 0.56 | 56.07 ± 0.57 | 64.02 ± 1.65 | 3.48 ± 1.98 | 28.0 | 0.0669 ± 0.0007 | 916224 |
| linear fusion of spikes, spikeDE LIF | 3 | 61.37 ± 0.61 | 56.35 ± 0.58 | 65.62 ± 0.24 | 4.25 ± 0.39 | 31.4 | 0.0692 ± 0.0025 | 1211904 |
| linear fusion of spikes + F xor R, spikeDE LIF | 3 | 62.43 ± 0.26 | 57.55 ± 0.46 | 65.95 ± 0.57 | 3.52 ± 0.83 | 33.2 | 0.0646 ± 0.0011 | 1359360 |
| linear fusion of charges + |H_f - H_r|, spikeDE LIF | 3 | 65.61 ± 1.11 | 59.57 ± 1.81 | 68.54 ± 1.86 | 2.92 ± 1.00 | 267.3 | 0.0548 ± 0.0003 | 1359360 |
| Neo-mixer OR, f-LIF | 3 | 59.75 ± 0.39 | 55.44 ± 1.24 | 63.66 ± 0.59 | 3.91 ± 0.94 | 32.8 | 0.0576 ± 0.0008 | 916224 |
| linear fusion of spikes, f-LIF | 3 | 60.58 ± 0.94 | 54.98 ± 1.02 | 63.81 ± 0.80 | 3.24 ± 1.21 | 36.2 | 0.0651 ± 0.0017 | 1211904 |
| linear fusion of spikes + F xor R, f-LIF | 3 | 60.20 ± 0.40 | 55.36 ± 0.66 | 64.06 ± 1.42 | 3.87 ± 1.20 | 37.9 | 0.0592 ± 0.0007 | 1359360 |
| linear fusion of charges + |H_f - H_r|, f-LIF | 3 | 62.18 ± 0.42 | 56.20 ± 0.56 | 66.77 ± 0.34 | 4.58 ± 0.71 | 271.8 | 0.0464 ± 0.0005 | 1359360 |
| Neo-mixer OR, population LIF (psi 0.5) | 3 | 63.06 ± 0.83 | 57.22 ± 1.49 | 65.45 ± 0.92 | 2.39 ± 1.61 | 36.5 | 0.0974 ± 0.0020 | 922368 |
| linear fusion of spikes, population LIF (psi 0.5) | 3 | 62.35 ± 1.65 | 56.76 ± 0.79 | 65.76 ± 1.29 | 3.40 ± 2.78 | 40.1 | 0.0614 ± 0.0002 | 1218048 |
| linear fusion of spikes + F xor R, population LIF (psi 0.5) | 3 | 61.44 ± 0.25 | 56.55 ± 0.85 | 65.14 ± 0.28 | 3.70 ± 0.45 | 42.4 | 0.0528 ± 0.0019 | 1365504 |
| linear fusion of charges + |H_f - H_r|, population LIF (psi 0.5) | 3 | 60.71 ± 1.55 | 56.79 ± 2.38 | 65.12 ± 1.44 | 4.42 ± 1.21 | 275.2 | 0.0388 ± 0.0012 | 1365504 |
| Neo-mixer OR, population f-LIF (psi 1) | 3 | 60.73 ± 1.67 | 55.74 ± 1.67 | 64.90 ± 0.85 | 4.17 ± 2.21 | 65.2 | 0.0873 ± 0.0021 | 922368 |
| linear fusion of spikes, population f-LIF (psi 1) | 3 | 62.13 ± 0.98 | 56.82 ± 2.37 | 66.02 ± 1.20 | 3.89 ± 0.41 | 68.8 | 0.0658 ± 0.0029 | 1218048 |
| linear fusion of spikes + F xor R, population f-LIF (psi 1) | 3 | 61.15 ± 0.53 | 56.78 ± 0.32 | 64.24 ± 2.00 | 3.09 ± 2.54 | 71.0 | 0.0564 ± 0.0012 | 1365504 |
| linear fusion of charges + |H_f - H_r|, population f-LIF (psi 1) | 3 | 61.14 ± 0.92 | 55.98 ± 0.54 | 64.35 ± 1.20 | 3.22 ± 0.98 | 303.5 | 0.0375 ± 0.0002 | 1365504 |

## Pre-registered comparisons (%p, paired by seed; better = mean >= +0.5 and A > B in >= 2 of 3 seeds)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| X1 + F xor R (spikes) | linear fusion of spikes + F xor R, spikeDE LIF | linear fusion of spikes, spikeDE LIF | +2.02 / +0.32 / +0.83 | +1.06 | better | 1.06 |  |
| X1 + F xor R (spikes) | linear fusion of spikes + F xor R, f-LIF | linear fusion of spikes, f-LIF | -1.01 / +0.33 / -0.46 | -0.38 | no clear difference | 1.05 |  |
| X1 + F xor R (spikes) | linear fusion of spikes + F xor R, population LIF (psi 0.5) | linear fusion of spikes, population LIF (psi 0.5) | -1.54 / -1.90 / +0.70 | -0.91 | worse | 1.06 |  |
| X1 + F xor R (spikes) | linear fusion of spikes + F xor R, population f-LIF (psi 1) | linear fusion of spikes, population f-LIF (psi 1) | -0.18 / -0.75 / -2.00 | -0.98 | worse | 1.03 |  |
| X2 spike xor vs charge difference | linear fusion of spikes + F xor R, spikeDE LIF | linear fusion of charges + |H_f - H_r|, spikeDE LIF | -2.70 / -4.61 / -2.24 | -3.18 | worse | 0.12 |  |
| X2 spike xor vs charge difference | linear fusion of spikes + F xor R, f-LIF | linear fusion of charges + |H_f - H_r|, f-LIF | -1.63 / -1.59 / -2.74 | -1.98 | worse | 0.14 |  |
| X2 spike xor vs charge difference | linear fusion of spikes + F xor R, population LIF (psi 0.5) | linear fusion of charges + |H_f - H_r|, population LIF (psi 0.5) | +2.65 / +0.13 / -0.59 | +0.73 | better | 0.15 |  |
| X2 spike xor vs charge difference | linear fusion of spikes + F xor R, population f-LIF (psi 1) | linear fusion of charges + |H_f - H_r|, population f-LIF (psi 1) | -0.93 / +1.68 / -0.71 | +0.01 | no clear difference | 0.23 |  |
| X3 vs OR | linear fusion of spikes + F xor R, spikeDE LIF | Neo-mixer OR, spikeDE LIF | +2.50 / +1.98 / +1.18 | +1.89 | better | 1.19 |  |
| X3 vs OR | linear fusion of spikes + F xor R, f-LIF | Neo-mixer OR, f-LIF | +0.47 / +0.75 / +0.12 | +0.45 | no clear difference | 1.15 |  |
| X3 vs OR | linear fusion of spikes + F xor R, population LIF (psi 0.5) | Neo-mixer OR, population LIF (psi 0.5) | -0.58 / -2.17 / -2.13 | -1.63 | worse | 1.16 |  |
| X3 vs OR | linear fusion of spikes + F xor R, population f-LIF (psi 1) | Neo-mixer OR, population f-LIF (psi 1) | +1.90 / +0.75 / -1.40 | +0.42 | no clear difference | 1.09 |  |
| X4 vs Spikformer (criterion 2) | linear fusion of spikes + F xor R, spikeDE LIF | Spikformer-style SSA (T=4) | -4.54 / -2.21 / -2.74 | -3.17 | worse | 0.26 | not met |
| X4 vs Spikformer (criterion 2) | linear fusion of spikes + F xor R, f-LIF | Spikformer-style SSA (T=4) | -6.63 / -4.27 / -5.29 | -5.40 | worse | 0.30 | not met |
| X4 vs Spikformer (criterion 2) | linear fusion of spikes + F xor R, population LIF (psi 0.5) | Spikformer-style SSA (T=4) | -5.68 / -2.83 / -3.96 | -4.16 | worse | 0.33 | not met |
| X4 vs Spikformer (criterion 2) | linear fusion of spikes + F xor R, population f-LIF (psi 1) | Spikformer-style SSA (T=4) | -6.42 / -2.65 / -4.27 | -4.45 | worse | 0.56 | not met |
