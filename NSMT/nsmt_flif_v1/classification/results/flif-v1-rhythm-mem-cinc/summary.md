# Membrane-fusion Neo-mixer `flif-v1-rhythm-mem-cinc` (CINC2017) — summary

Best-validation weights on the test set; CinC score and macro-F1; mean ± sd over seeds (42, 7, 2026) (%). Energy counts the real-valued fusion inputs as MAC (syops: AC only for 0/1 inputs); attention products are not counted, as before.

| condition | n | score | macro-F1 | best val | val − test | total µJ (FFT) | neocortex firing | params |
|---|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 3 | 65.60 ± 1.48 | 61.94 ± 0.41 | 68.79 ± 2.19 | 3.19 ± 0.76 | 127.4 | n/a | 918088 |
| Neo-mixer OR, spikeDE LIF | 3 | 60.54 ± 0.56 | 56.07 ± 0.57 | 64.02 ± 1.65 | 3.48 ± 1.98 | 28.0 | 0.0669 ± 0.0007 | 916224 |
| linear fusion of spikes, spikeDE LIF | 3 | 61.37 ± 0.61 | 56.35 ± 0.58 | 65.62 ± 0.24 | 4.25 ± 0.39 | 31.4 | 0.0692 ± 0.0025 | 1211904 |
| linear fusion of charges, spikeDE LIF | 3 | 61.03 ± 1.26 | 56.63 ± 1.87 | 65.38 ± 1.10 | 4.35 ± 0.31 | 187.2 | 0.0460 ± 0.0021 | 1211904 |
| linear fusion of charges + |H_f - H_r|, spikeDE LIF | 3 | 65.61 ± 1.11 | 59.57 ± 1.81 | 68.54 ± 1.86 | 2.92 ± 1.00 | 267.3 | 0.0548 ± 0.0003 | 1359360 |
| BiX (spike q / k), spikeDE LIF | 3 | 61.50 ± 1.24 | 56.90 ± 0.90 | 66.39 ± 1.09 | 4.89 ± 2.30 | 35.8 | 0.0630 ± 0.0014 | 1511752 |
| BiX with charge q / k, spikeDE LIF | 3 | 60.08 ± 1.74 | 53.01 ± 0.37 | 62.88 ± 2.65 | 2.80 ± 3.20 | 191.3 | 0.0612 ± 0.0058 | 1511752 |
| Neo-mixer OR, f-LIF | 3 | 59.75 ± 0.39 | 55.44 ± 1.24 | 63.66 ± 0.59 | 3.91 ± 0.94 | 32.8 | 0.0576 ± 0.0008 | 916224 |
| linear fusion of spikes, f-LIF | 3 | 60.58 ± 0.94 | 54.98 ± 1.02 | 63.81 ± 0.80 | 3.24 ± 1.21 | 36.2 | 0.0651 ± 0.0017 | 1211904 |
| linear fusion of charges, f-LIF | 3 | 59.13 ± 1.47 | 54.16 ± 1.28 | 62.91 ± 0.93 | 3.78 ± 2.10 | 191.9 | 0.0439 ± 0.0027 | 1211904 |
| linear fusion of charges + |H_f - H_r|, f-LIF | 3 | 62.18 ± 0.42 | 56.20 ± 0.56 | 66.77 ± 0.34 | 4.58 ± 0.71 | 271.8 | 0.0464 ± 0.0005 | 1359360 |
| BiX (spike q / k), f-LIF | 3 | 61.11 ± 1.10 | 56.27 ± 1.55 | 66.66 ± 1.67 | 5.55 ± 1.93 | 40.1 | 0.0580 ± 0.0007 | 1511752 |
| BiX with charge q / k, f-LIF | 3 | 59.87 ± 1.88 | 52.57 ± 3.04 | 61.61 ± 1.73 | 1.74 ± 1.52 | 195.8 | 0.0531 ± 0.0066 | 1511752 |
| Neo-mixer OR, population LIF (psi 0.5) | 3 | 63.06 ± 0.83 | 57.22 ± 1.49 | 65.45 ± 0.92 | 2.39 ± 1.61 | 36.5 | 0.0974 ± 0.0020 | 922368 |
| linear fusion of spikes, population LIF (psi 0.5) | 3 | 62.35 ± 1.65 | 56.76 ± 0.79 | 65.76 ± 1.29 | 3.40 ± 2.78 | 40.1 | 0.0614 ± 0.0002 | 1218048 |
| linear fusion of charges, population LIF (psi 0.5) | 3 | 61.92 ± 1.39 | 56.38 ± 2.48 | 64.91 ± 0.50 | 2.99 ± 1.53 | 195.9 | 0.0292 ± 0.0003 | 1218048 |
| linear fusion of charges + |H_f - H_r|, population LIF (psi 0.5) | 3 | 60.71 ± 1.55 | 56.79 ± 2.38 | 65.12 ± 1.44 | 4.42 ± 1.21 | 275.2 | 0.0388 ± 0.0012 | 1365504 |
| BiX (spike q / k), population LIF (psi 0.5) | 3 | 61.93 ± 0.02 | 56.82 ± 0.70 | 64.54 ± 0.92 | 2.62 ± 0.91 | 45.7 | 0.0556 ± 0.0039 | 1517896 |
| BiX with charge q / k, population LIF (psi 0.5) | 3 | 56.01 ± 3.70 | 50.77 ± 3.09 | 59.69 ± 3.39 | 3.68 ± 0.82 | 199.1 | 0.0443 ± 0.0018 | 1517896 |
| Neo-mixer OR, population f-LIF (psi 1) | 3 | 60.73 ± 1.67 | 55.74 ± 1.67 | 64.90 ± 0.85 | 4.17 ± 2.21 | 65.2 | 0.0873 ± 0.0021 | 922368 |
| linear fusion of spikes, population f-LIF (psi 1) | 3 | 62.13 ± 0.98 | 56.82 ± 2.37 | 66.02 ± 1.20 | 3.89 ± 0.41 | 68.8 | 0.0658 ± 0.0029 | 1218048 |
| linear fusion of charges, population f-LIF (psi 1) | 3 | 60.85 ± 0.64 | 55.58 ± 0.94 | 63.52 ± 1.25 | 2.67 ± 1.53 | 224.1 | 0.0320 ± 0.0012 | 1218048 |
| linear fusion of charges + |H_f - H_r|, population f-LIF (psi 1) | 3 | 61.14 ± 0.92 | 55.98 ± 0.54 | 64.35 ± 1.20 | 3.22 ± 0.98 | 303.5 | 0.0375 ± 0.0002 | 1365504 |
| BiX (spike q / k), population f-LIF (psi 1) | 3 | 59.62 ± 0.34 | 54.73 ± 2.23 | 62.75 ± 0.78 | 3.13 ± 0.68 | 73.6 | 0.0574 ± 0.0017 | 1517896 |
| BiX with charge q / k, population f-LIF (psi 1) | 3 | 55.03 ± 1.13 | 49.72 ± 0.86 | 57.52 ± 1.61 | 2.49 ± 2.45 | 227.7 | 0.0546 ± 0.0056 | 1517896 |

## Pre-registered comparisons (%p, paired by seed; better = mean >= +0.5 and A > B in >= 2 of 3 seeds)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| M1 charges vs spikes (linear fusion) | linear fusion of charges, spikeDE LIF | linear fusion of spikes, spikeDE LIF | -0.24 / -1.71 / +0.93 | -0.34 | no clear difference | 5.96 |  |
| M1 charges vs spikes (linear fusion) | linear fusion of charges, f-LIF | linear fusion of spikes, f-LIF | -2.43 / +0.76 / -2.66 | -1.45 | worse | 5.31 |  |
| M1 charges vs spikes (linear fusion) | linear fusion of charges, population LIF (psi 0.5) | linear fusion of spikes, population LIF (psi 0.5) | -1.43 / -0.09 / +0.21 | -0.44 | no clear difference | 4.88 |  |
| M1 charges vs spikes (linear fusion) | linear fusion of charges, population f-LIF (psi 1) | linear fusion of spikes, population f-LIF (psi 1) | -0.52 / -2.05 / -1.26 | -1.28 | worse | 3.26 |  |
| M2 + |H_f - H_r| | linear fusion of charges + |H_f - H_r|, spikeDE LIF | linear fusion of charges, spikeDE LIF | +4.96 / +6.64 / +2.14 | +4.58 | better | 1.43 |  |
| M2 + |H_f - H_r| | linear fusion of charges + |H_f - H_r|, f-LIF | linear fusion of charges, f-LIF | +3.05 / +1.16 / +4.94 | +3.05 | better | 1.42 |  |
| M2 + |H_f - H_r| | linear fusion of charges + |H_f - H_r|, population LIF (psi 0.5) | linear fusion of charges, population LIF (psi 0.5) | -2.76 / -1.95 / +1.08 | -1.21 | worse | 1.41 |  |
| M2 + |H_f - H_r| | linear fusion of charges + |H_f - H_r|, population f-LIF (psi 1) | linear fusion of charges, population f-LIF (psi 1) | +1.28 / -0.38 / -0.03 | +0.29 | no clear difference | 1.35 |  |
| M3 charges + |H_f - H_r| vs spikes | linear fusion of charges + |H_f - H_r|, spikeDE LIF | linear fusion of spikes, spikeDE LIF | +4.72 / +4.93 / +3.07 | +4.24 | better | 8.51 |  |
| M3 charges + |H_f - H_r| vs spikes | linear fusion of charges + |H_f - H_r|, f-LIF | linear fusion of spikes, f-LIF | +0.62 / +1.92 / +2.28 | +1.61 | better | 7.52 |  |
| M3 charges + |H_f - H_r| vs spikes | linear fusion of charges + |H_f - H_r|, population LIF (psi 0.5) | linear fusion of spikes, population LIF (psi 0.5) | -4.19 / -2.04 / +1.29 | -1.65 | worse | 6.86 |  |
| M3 charges + |H_f - H_r| vs spikes | linear fusion of charges + |H_f - H_r|, population f-LIF (psi 1) | linear fusion of spikes, population f-LIF (psi 1) | +0.75 / -2.43 / -1.29 | -0.99 | worse | 4.41 |  |
| M4 charge q / k vs spike q / k (BiX) | BiX with charge q / k, spikeDE LIF | BiX (spike q / k), spikeDE LIF | -2.99 / +1.50 / -2.79 | -1.43 | worse | 5.34 |  |
| M4 charge q / k vs spike q / k (BiX) | BiX with charge q / k, f-LIF | BiX (spike q / k), f-LIF | -1.67 / +1.74 / -3.80 | -1.24 | worse | 4.88 |  |
| M4 charge q / k vs spike q / k (BiX) | BiX with charge q / k, population LIF (psi 0.5) | BiX (spike q / k), population LIF (psi 0.5) | -2.14 / -9.51 / -6.11 | -5.92 | worse | 4.35 |  |
| M4 charge q / k vs spike q / k (BiX) | BiX with charge q / k, population f-LIF (psi 1) | BiX (spike q / k), population f-LIF (psi 1) | -3.76 / -5.33 / -4.69 | -4.59 | worse | 3.09 |  |
| M5 vs OR | linear fusion of charges, spikeDE LIF | Neo-mixer OR, spikeDE LIF | +0.25 / -0.05 / +1.28 | +0.49 | no clear difference | 6.69 |  |
| M5 vs OR | linear fusion of charges + |H_f - H_r|, spikeDE LIF | Neo-mixer OR, spikeDE LIF | +5.20 / +6.59 / +3.42 | +5.07 | better | 9.55 |  |
| M5 vs OR | BiX with charge q / k, spikeDE LIF | Neo-mixer OR, spikeDE LIF | -0.35 / +1.69 / -2.74 | -0.46 | no clear difference | 6.84 |  |
| M5 vs OR | linear fusion of charges, f-LIF | Neo-mixer OR, f-LIF | -0.96 / +1.18 / -2.08 | -0.62 | worse | 5.84 |  |
| M5 vs OR | linear fusion of charges + |H_f - H_r|, f-LIF | Neo-mixer OR, f-LIF | +2.09 / +2.34 / +2.86 | +2.43 | better | 8.28 |  |
| M5 vs OR | BiX with charge q / k, f-LIF | Neo-mixer OR, f-LIF | +0.01 / +2.20 / -1.86 | +0.12 | no clear difference | 5.96 |  |
| M5 vs OR | linear fusion of charges, population LIF (psi 0.5) | Neo-mixer OR, population LIF (psi 0.5) | -0.47 / -0.35 / -2.62 | -1.15 | worse | 5.36 |  |
| M5 vs OR | linear fusion of charges + |H_f - H_r|, population LIF (psi 0.5) | Neo-mixer OR, population LIF (psi 0.5) | -3.23 / -2.30 / -1.55 | -2.36 | worse | 7.53 |  |
| M5 vs OR | BiX with charge q / k, population LIF (psi 0.5) | Neo-mixer OR, population LIF (psi 0.5) | -2.34 / -11.35 / -7.49 | -7.06 | worse | 5.45 |  |
| M5 vs OR | linear fusion of charges, population f-LIF (psi 1) | Neo-mixer OR, population f-LIF (psi 1) | +1.55 / -0.55 / -0.66 | +0.12 | no clear difference | 3.44 |  |
| M5 vs OR | linear fusion of charges + |H_f - H_r|, population f-LIF (psi 1) | Neo-mixer OR, population f-LIF (psi 1) | +2.83 / -0.93 / -0.69 | +0.41 | no clear difference | 4.66 |  |
| M5 vs OR | BiX with charge q / k, population f-LIF (psi 1) | Neo-mixer OR, population f-LIF (psi 1) | -2.70 / -7.03 / -7.37 | -5.70 | worse | 3.49 |  |
| M6 vs Spikformer (criterion 2) | linear fusion of charges, spikeDE LIF | Spikformer-style SSA (T=4) | -6.80 / -4.24 / -2.64 | -4.56 | worse | 1.47 | not met |
| M6 vs Spikformer (criterion 2) | linear fusion of charges + |H_f - H_r|, spikeDE LIF | Spikformer-style SSA (T=4) | -1.84 / +2.40 / -0.50 | +0.02 | no clear difference | 2.10 | not met |
| M6 vs Spikformer (criterion 2) | BiX with charge q / k, spikeDE LIF | Spikformer-style SSA (T=4) | -7.39 / -2.50 / -6.67 | -5.52 | worse | 1.50 | not met |
| M6 vs Spikformer (criterion 2) | linear fusion of charges, f-LIF | Spikformer-style SSA (T=4) | -8.06 / -3.84 / -7.49 | -6.46 | worse | 1.51 | not met |
| M6 vs Spikformer (criterion 2) | linear fusion of charges + |H_f - H_r|, f-LIF | Spikformer-style SSA (T=4) | -5.00 / -2.68 / -2.55 | -3.41 | worse | 2.13 | not met |
| M6 vs Spikformer (criterion 2) | BiX with charge q / k, f-LIF | Spikformer-style SSA (T=4) | -7.09 / -2.83 / -7.27 | -5.73 | worse | 1.54 | not met |
| M6 vs Spikformer (criterion 2) | linear fusion of charges, population LIF (psi 0.5) | Spikformer-style SSA (T=4) | -5.58 / -1.01 / -4.46 | -3.68 | worse | 1.54 | not met |
| M6 vs Spikformer (criterion 2) | linear fusion of charges + |H_f - H_r|, population LIF (psi 0.5) | Spikformer-style SSA (T=4) | -8.33 / -2.96 / -3.38 | -4.89 | worse | 2.16 | not met |
| M6 vs Spikformer (criterion 2) | BiX with charge q / k, population LIF (psi 0.5) | Spikformer-style SSA (T=4) | -7.45 / -12.01 / -9.32 | -9.59 | worse | 1.56 | not met |
| M6 vs Spikformer (criterion 2) | linear fusion of charges, population f-LIF (psi 1) | Spikformer-style SSA (T=4) | -6.77 / -3.95 / -3.54 | -4.75 | worse | 1.76 | not met |
| M6 vs Spikformer (criterion 2) | linear fusion of charges + |H_f - H_r|, population f-LIF (psi 1) | Spikformer-style SSA (T=4) | -5.49 / -4.33 / -3.56 | -4.46 | worse | 2.38 | not met |
| M6 vs Spikformer (criterion 2) | BiX with charge q / k, population f-LIF (psi 1) | Spikformer-style SSA (T=4) | -11.02 / -10.43 / -10.24 | -10.57 | worse | 1.79 | not met |
