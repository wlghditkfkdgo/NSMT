# Spiking-difference fusion Neo-mixer `flif-v1-rhythm-sd-cinc` (CINC2017) — summary

Best-validation weights on the test set; CinC fold 0; mean ± sd over seeds (42, 7, 2026) (%). The fusion input is 0/1 (AC).

| condition | n | score | macro-F1 | best val | val − test | total µJ (FFT) | neocortex firing | params |
|---|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 3 | 65.60 ± 1.48 | 61.94 ± 0.41 | 68.79 ± 2.19 | 3.19 ± 0.76 | 127.4 | n/a | 918088 |
| Neo-mixer OR, spikeDE LIF | 3 | 60.54 ± 0.56 | 56.07 ± 0.57 | 64.02 ± 1.65 | 3.48 ± 1.98 | 28.0 | 0.0669 ± 0.0007 | 916224 |
| spike linear fusion, spikeDE LIF | 3 | 61.37 ± 0.61 | 56.35 ± 0.58 | 65.62 ± 0.24 | 4.25 ± 0.39 | 31.4 | 0.0692 ± 0.0025 | 1211904 |
| spikes + F xor R, spikeDE LIF | 3 | 62.43 ± 0.26 | 57.55 ± 0.46 | 65.95 ± 0.57 | 3.52 ± 0.83 | 33.2 | 0.0646 ± 0.0011 | 1359360 |
| spikes + LIF(BN(|H_f - H_r|)), spikeDE LIF | 3 | 64.98 ± 0.41 | 59.21 ± 0.21 | 69.44 ± 0.27 | 4.46 ± 0.44 | 33.9 | 0.0671 ± 0.0005 | 1360128 |
| charges + |H_f - H_r| (3), spikeDE LIF | 3 | 65.61 ± 1.11 | 59.57 ± 1.81 | 68.54 ± 1.86 | 2.92 ± 1.00 | 267.3 | 0.0548 ± 0.0003 | 1359360 |
| Neo-mixer OR, f-LIF | 3 | 59.75 ± 0.39 | 55.44 ± 1.24 | 63.66 ± 0.59 | 3.91 ± 0.94 | 32.8 | 0.0576 ± 0.0008 | 916224 |
| spike linear fusion, f-LIF | 3 | 60.58 ± 0.94 | 54.98 ± 1.02 | 63.81 ± 0.80 | 3.24 ± 1.21 | 36.2 | 0.0651 ± 0.0017 | 1211904 |
| spikes + F xor R, f-LIF | 3 | 60.20 ± 0.40 | 55.36 ± 0.66 | 64.06 ± 1.42 | 3.87 ± 1.20 | 37.9 | 0.0592 ± 0.0007 | 1359360 |
| spikes + LIF(BN(|H_f - H_r|)), f-LIF | 3 | 62.78 ± 0.71 | 57.88 ± 1.12 | 66.67 ± 0.60 | 3.89 ± 1.31 | 38.3 | 0.0567 ± 0.0009 | 1360128 |
| charges + |H_f - H_r| (3), f-LIF | 3 | 62.18 ± 0.42 | 56.20 ± 0.56 | 66.77 ± 0.34 | 4.58 ± 0.71 | 271.8 | 0.0464 ± 0.0005 | 1359360 |
| Neo-mixer OR, population LIF (psi 0.5) | 3 | 63.06 ± 0.83 | 57.22 ± 1.49 | 65.45 ± 0.92 | 2.39 ± 1.61 | 36.5 | 0.0974 ± 0.0020 | 922368 |
| spike linear fusion, population LIF (psi 0.5) | 3 | 62.35 ± 1.65 | 56.76 ± 0.79 | 65.76 ± 1.29 | 3.40 ± 2.78 | 40.1 | 0.0614 ± 0.0002 | 1218048 |
| spikes + F xor R, population LIF (psi 0.5) | 3 | 61.44 ± 0.25 | 56.55 ± 0.85 | 65.14 ± 0.28 | 3.70 ± 0.45 | 42.4 | 0.0528 ± 0.0019 | 1365504 |
| spikes + LIF(BN(|H_f - H_r|)), population LIF (psi 0.5) | 3 | 61.25 ± 1.17 | 55.85 ± 0.31 | 66.09 ± 0.87 | 4.84 ± 2.01 | 42.5 | 0.0627 ± 0.0016 | 1366272 |
| charges + |H_f - H_r| (3), population LIF (psi 0.5) | 3 | 60.71 ± 1.55 | 56.79 ± 2.38 | 65.12 ± 1.44 | 4.42 ± 1.21 | 275.2 | 0.0388 ± 0.0012 | 1365504 |
| Neo-mixer OR, population f-LIF (psi 1) | 3 | 60.73 ± 1.67 | 55.74 ± 1.67 | 64.90 ± 0.85 | 4.17 ± 2.21 | 65.2 | 0.0873 ± 0.0021 | 922368 |
| spike linear fusion, population f-LIF (psi 1) | 3 | 62.13 ± 0.98 | 56.82 ± 2.37 | 66.02 ± 1.20 | 3.89 ± 0.41 | 68.8 | 0.0658 ± 0.0029 | 1218048 |
| spikes + F xor R, population f-LIF (psi 1) | 3 | 61.15 ± 0.53 | 56.78 ± 0.32 | 64.24 ± 2.00 | 3.09 ± 2.54 | 71.0 | 0.0564 ± 0.0012 | 1365504 |
| spikes + LIF(BN(|H_f - H_r|)), population f-LIF (psi 1) | 3 | 61.24 ± 1.64 | 57.00 ± 1.33 | 65.82 ± 0.75 | 4.58 ± 2.32 | 71.3 | 0.0650 ± 0.0014 | 1366272 |
| charges + |H_f - H_r| (3), population f-LIF (psi 1) | 3 | 61.14 ± 0.92 | 55.98 ± 0.54 | 64.35 ± 1.20 | 3.22 ± 0.98 | 303.5 | 0.0375 ± 0.0002 | 1365504 |
| Neo-mixer OR, soma-less tau groups, LIF | 3 | 59.27 ± 0.45 | 55.01 ± 1.30 | 61.30 ± 0.83 | 2.02 ± 0.74 | 26.0 | 0.1023 ± 0.0021 | 916224 |
| spike linear fusion, soma-less tau groups, LIF | 3 | 57.19 ± 0.07 | 52.53 ± 1.02 | 59.78 ± 0.42 | 2.59 ± 0.50 | 29.6 | 0.0863 ± 0.0012 | 1211904 |
| spikes + F xor R, soma-less tau groups, LIF | 3 | 58.06 ± 0.20 | 54.08 ± 0.20 | 59.28 ± 2.03 | 1.22 ± 2.00 | 31.9 | 0.0834 ± 0.0014 | 1359360 |
| spikes + LIF(BN(|H_f - H_r|)), soma-less tau groups, LIF | 3 | 58.20 ± 1.11 | 53.28 ± 2.71 | 60.47 ± 1.46 | 2.27 ± 0.38 | 31.7 | 0.0867 ± 0.0014 | 1360128 |
| charges + |H_f - H_r| (3), soma-less tau groups, LIF | 3 | 58.62 ± 1.07 | 55.04 ± 1.91 | 59.58 ± 2.00 | 0.96 ± 1.01 | 264.8 | 0.0795 ± 0.0004 | 1359360 |
| Neo-mixer OR, soma-less tau groups, f-LIF | 3 | 59.51 ± 1.81 | 54.34 ± 0.37 | 60.22 ± 1.40 | 0.70 ± 1.15 | 30.5 | 0.0872 ± 0.0034 | 916224 |
| spike linear fusion, soma-less tau groups, f-LIF | 3 | 57.41 ± 0.30 | 52.77 ± 1.11 | 60.38 ± 0.37 | 2.97 ± 0.60 | 34.0 | 0.0759 ± 0.0019 | 1211904 |
| spikes + F xor R, soma-less tau groups, f-LIF | 3 | 57.73 ± 0.37 | 53.47 ± 1.38 | 61.69 ± 1.06 | 3.96 ± 1.25 | 36.4 | 0.0745 ± 0.0008 | 1359360 |
| spikes + LIF(BN(|H_f - H_r|)), soma-less tau groups, f-LIF | 3 | 58.68 ± 0.45 | 54.75 ± 0.68 | 58.55 ± 0.44 | -0.12 ± 0.89 | 36.6 | 0.0805 ± 0.0025 | 1360128 |
| charges + |H_f - H_r| (3), soma-less tau groups, f-LIF | 3 | 57.31 ± 1.12 | 53.12 ± 0.56 | 59.77 ± 0.74 | 2.46 ± 1.12 | 269.2 | 0.0743 ± 0.0003 | 1359360 |

## Pre-registered comparisons (%p, paired by seed; better = mean >= +0.5 and A > B in >= 2 of 3 seeds)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| SD1 vs spike linear fusion | spikes + LIF(BN(|H_f - H_r|)), spikeDE LIF | spike linear fusion, spikeDE LIF | +3.89 / +3.06 / +3.86 | +3.60 | better | 1.08 |  |
| SD1 vs spike linear fusion | spikes + LIF(BN(|H_f - H_r|)), f-LIF | spike linear fusion, f-LIF | +0.97 / +2.36 / +3.27 | +2.20 | better | 1.06 |  |
| SD1 vs spike linear fusion | spikes + LIF(BN(|H_f - H_r|)), population LIF (psi 0.5) | spike linear fusion, population LIF (psi 0.5) | -1.19 / -1.58 / -0.56 | -1.11 | worse | 1.06 |  |
| SD1 vs spike linear fusion | spikes + LIF(BN(|H_f - H_r|)), population f-LIF (psi 1) | spike linear fusion, population f-LIF (psi 1) | -1.61 / -0.76 / -0.28 | -0.88 | worse | 1.04 |  |
| SD1 vs spike linear fusion | spikes + LIF(BN(|H_f - H_r|)), soma-less tau groups, LIF | spike linear fusion, soma-less tau groups, LIF | +2.15 / +1.03 / -0.15 | +1.01 | better | 1.07 |  |
| SD1 vs spike linear fusion | spikes + LIF(BN(|H_f - H_r|)), soma-less tau groups, f-LIF | spike linear fusion, soma-less tau groups, f-LIF | +1.44 / +1.30 / +1.07 | +1.27 | better | 1.08 |  |
| SD2 vs spike xor | spikes + LIF(BN(|H_f - H_r|)), spikeDE LIF | spikes + F xor R, spikeDE LIF | +1.87 / +2.74 / +3.03 | +2.55 | better | 1.02 |  |
| SD2 vs spike xor | spikes + LIF(BN(|H_f - H_r|)), f-LIF | spikes + F xor R, f-LIF | +1.97 / +2.03 / +3.73 | +2.58 | better | 1.01 |  |
| SD2 vs spike xor | spikes + LIF(BN(|H_f - H_r|)), population LIF (psi 0.5) | spikes + F xor R, population LIF (psi 0.5) | +0.35 / +0.33 / -1.26 | -0.19 | no clear difference | 1.00 |  |
| SD2 vs spike xor | spikes + LIF(BN(|H_f - H_r|)), population f-LIF (psi 1) | spikes + F xor R, population f-LIF (psi 1) | -1.43 / -0.01 / +1.72 | +0.09 | no clear difference | 1.00 |  |
| SD2 vs spike xor | spikes + LIF(BN(|H_f - H_r|)), soma-less tau groups, LIF | spikes + F xor R, soma-less tau groups, LIF | +1.50 / +0.06 / -1.12 | +0.15 | no clear difference | 0.99 |  |
| SD2 vs spike xor | spikes + LIF(BN(|H_f - H_r|)), soma-less tau groups, f-LIF | spikes + F xor R, soma-less tau groups, f-LIF | +1.09 / +0.92 / +0.82 | +0.95 | better | 1.00 |  |
| SD3 vs charge difference (3) | spikes + LIF(BN(|H_f - H_r|)), spikeDE LIF | charges + |H_f - H_r| (3), spikeDE LIF | -0.83 / -1.87 / +0.79 | -0.64 | worse | 0.13 |  |
| SD3 vs charge difference (3) | spikes + LIF(BN(|H_f - H_r|)), f-LIF | charges + |H_f - H_r| (3), f-LIF | +0.35 / +0.44 / +0.99 | +0.59 | better | 0.14 |  |
| SD3 vs charge difference (3) | spikes + LIF(BN(|H_f - H_r|)), population LIF (psi 0.5) | charges + |H_f - H_r| (3), population LIF (psi 0.5) | +3.00 / +0.46 / -1.84 | +0.54 | better | 0.15 |  |
| SD3 vs charge difference (3) | spikes + LIF(BN(|H_f - H_r|)), population f-LIF (psi 1) | charges + |H_f - H_r| (3), population f-LIF (psi 1) | -2.36 / +1.67 / +1.01 | +0.10 | no clear difference | 0.23 |  |
| SD3 vs charge difference (3) | spikes + LIF(BN(|H_f - H_r|)), soma-less tau groups, LIF | charges + |H_f - H_r| (3), soma-less tau groups, LIF | +1.53 / -1.69 / -1.10 | -0.42 | no clear difference | 0.12 |  |
| SD3 vs charge difference (3) | spikes + LIF(BN(|H_f - H_r|)), soma-less tau groups, f-LIF | charges + |H_f - H_r| (3), soma-less tau groups, f-LIF | +2.29 / +2.25 / -0.44 | +1.37 | better | 0.14 |  |
| SD4 vs OR | spikes + LIF(BN(|H_f - H_r|)), spikeDE LIF | Neo-mixer OR, spikeDE LIF | +4.37 / +4.72 / +4.21 | +4.43 | better | 1.21 |  |
| SD4 vs OR | spikes + LIF(BN(|H_f - H_r|)), f-LIF | Neo-mixer OR, f-LIF | +2.44 / +2.78 / +3.85 | +3.02 | better | 1.17 |  |
| SD4 vs OR | spikes + LIF(BN(|H_f - H_r|)), population LIF (psi 0.5) | Neo-mixer OR, population LIF (psi 0.5) | -0.23 / -1.84 / -3.39 | -1.82 | worse | 1.16 |  |
| SD4 vs OR | spikes + LIF(BN(|H_f - H_r|)), population f-LIF (psi 1) | Neo-mixer OR, population f-LIF (psi 1) | +0.47 / +0.74 / +0.32 | +0.51 | better | 1.09 |  |
| SD4 vs OR | spikes + LIF(BN(|H_f - H_r|)), soma-less tau groups, LIF | Neo-mixer OR, soma-less tau groups, LIF | -0.39 / -0.69 / -2.13 | -1.07 | worse | 1.22 |  |
| SD4 vs OR | spikes + LIF(BN(|H_f - H_r|)), soma-less tau groups, f-LIF | Neo-mixer OR, soma-less tau groups, f-LIF | +1.38 / -1.04 / -2.85 | -0.84 | worse | 1.20 |  |
| SD5 vs Spikformer (criterion 2) | spikes + LIF(BN(|H_f - H_r|)), spikeDE LIF | Spikformer-style SSA (T=4) | -2.67 / +0.53 / +0.28 | -0.62 | no clear difference | 0.27 | not met |
| SD5 vs Spikformer (criterion 2) | spikes + LIF(BN(|H_f - H_r|)), f-LIF | Spikformer-style SSA (T=4) | -4.66 / -2.25 / -1.56 | -2.82 | worse | 0.30 | not met |
| SD5 vs Spikformer (criterion 2) | spikes + LIF(BN(|H_f - H_r|)), population LIF (psi 0.5) | Spikformer-style SSA (T=4) | -5.33 / -2.50 / -5.22 | -4.35 | worse | 0.33 | not met |
| SD5 vs Spikformer (criterion 2) | spikes + LIF(BN(|H_f - H_r|)), population f-LIF (psi 1) | Spikformer-style SSA (T=4) | -7.85 / -2.66 / -2.55 | -4.36 | worse | 0.56 | not met |
| SD5 vs Spikformer (criterion 2) | spikes + LIF(BN(|H_f - H_r|)), soma-less tau groups, LIF | Spikformer-style SSA (T=4) | -7.91 / -6.27 / -8.00 | -7.39 | worse | 0.25 | not met |
| SD5 vs Spikformer (criterion 2) | spikes + LIF(BN(|H_f - H_r|)), soma-less tau groups, f-LIF | Spikformer-style SSA (T=4) | -8.37 / -5.43 / -6.95 | -6.92 | worse | 0.29 | not met |
