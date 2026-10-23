# Linear-fusion Neo-mixer `flif-v1-rhythm-lin-cinc` (CINC2017) — summary

Best-validation weights on the test set; CinC score; mean ± sd over seeds (42, 7, 2026) (%).

| condition | n | score | best val | val − test | total µJ (FFT) | neocortex firing | params |
|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 3 | 65.60 ± 1.48 | 68.79 ± 2.19 | 3.19 ± 0.76 | 127.4 | n/a | 918088 |
| Spikformer-style SSA + attention BN | 3 | 65.40 ± 0.57 | 71.06 ± 1.05 | 5.65 ± 0.54 | 125.0 | n/a | 918856 |
| no mixing (floor) | 3 | 58.20 ± 1.94 | 59.74 ± 0.40 | 1.55 ± 2.16 | 18.7 | n/a | 323328 |
| Neo-mixer OR, spikeDE LIF | 3 | 60.54 ± 0.56 | 64.02 ± 1.65 | 3.48 ± 1.98 | 28.0 | 0.0669 ± 0.0007 | 916224 |
| Neo-mixer linear fusion, spikeDE LIF | 3 | 61.37 ± 0.61 | 65.62 ± 0.24 | 4.25 ± 0.39 | 31.4 | 0.0692 ± 0.0025 | 1211904 |
| BiX, spikeDE LIF | 3 | 61.50 ± 1.24 | 66.39 ± 1.09 | 4.89 ± 2.30 | 35.8 | 0.0630 ± 0.0014 | 1511752 |
| Neo-mixer OR, f-LIF | 3 | 59.75 ± 0.39 | 63.66 ± 0.59 | 3.91 ± 0.94 | 32.8 | 0.0576 ± 0.0008 | 916224 |
| Neo-mixer linear fusion, f-LIF | 3 | 60.58 ± 0.94 | 63.81 ± 0.80 | 3.24 ± 1.21 | 36.2 | 0.0651 ± 0.0017 | 1211904 |
| BiX, f-LIF | 3 | 61.11 ± 1.10 | 66.66 ± 1.67 | 5.55 ± 1.93 | 40.1 | 0.0580 ± 0.0007 | 1511752 |
| Neo-mixer OR, population LIF (psi 0.5) | 3 | 63.06 ± 0.83 | 65.45 ± 0.92 | 2.39 ± 1.61 | 36.5 | 0.0974 ± 0.0020 | 922368 |
| Neo-mixer linear fusion, population LIF (psi 0.5) | 3 | 62.35 ± 1.65 | 65.76 ± 1.29 | 3.40 ± 2.78 | 40.1 | 0.0614 ± 0.0002 | 1218048 |
| BiX, population LIF (psi 0.5) | 3 | 61.93 ± 0.02 | 64.54 ± 0.92 | 2.62 ± 0.91 | 45.7 | 0.0556 ± 0.0039 | 1517896 |
| Neo-mixer OR, population f-LIF (psi 1) | 3 | 60.73 ± 1.67 | 64.90 ± 0.85 | 4.17 ± 2.21 | 65.2 | 0.0873 ± 0.0021 | 922368 |
| Neo-mixer linear fusion, population f-LIF (psi 1) | 3 | 62.13 ± 0.98 | 66.02 ± 1.20 | 3.89 ± 0.41 | 68.8 | 0.0658 ± 0.0029 | 1218048 |
| BiX, population f-LIF (psi 1) | 3 | 59.62 ± 0.34 | 62.75 ± 0.78 | 3.13 ± 0.68 | 73.6 | 0.0574 ± 0.0017 | 1517896 |

## Pre-registered comparisons (%p, paired by seed)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| L1 OR -> linear fusion | Neo-mixer linear fusion, spikeDE LIF | Neo-mixer OR, spikeDE LIF | +0.48 / +1.66 / +0.35 | +0.83 | better | 1.12 |  |
| L1 OR -> linear fusion | Neo-mixer linear fusion, f-LIF | Neo-mixer OR, f-LIF | +1.47 / +0.42 / +0.58 | +0.82 | better | 1.10 |  |
| L1 OR -> linear fusion | Neo-mixer linear fusion, population LIF (psi 0.5) | Neo-mixer OR, population LIF (psi 0.5) | +0.96 / -0.26 / -2.83 | -0.71 | worse | 1.10 |  |
| L1 OR -> linear fusion | Neo-mixer linear fusion, population f-LIF (psi 1) | Neo-mixer OR, population f-LIF (psi 1) | +2.08 / +1.51 / +0.60 | +1.40 | better | 1.06 |  |
| L2 linear fusion vs BiX | Neo-mixer linear fusion, spikeDE LIF | BiX, spikeDE LIF | -2.16 / +1.47 / +0.30 | -0.13 | no clear difference | 0.88 |  |
| L2 linear fusion vs BiX | Neo-mixer linear fusion, f-LIF | BiX, f-LIF | -0.20 / -0.04 / -1.36 | -0.53 | worse | 0.90 |  |
| L2 linear fusion vs BiX | Neo-mixer linear fusion, population LIF (psi 0.5) | BiX, population LIF (psi 0.5) | +1.16 / +1.57 / -1.46 | +0.43 | no clear difference | 0.88 |  |
| L2 linear fusion vs BiX | Neo-mixer linear fusion, population f-LIF (psi 1) | BiX, population f-LIF (psi 1) | +1.02 / +3.21 / +3.28 | +2.51 | better | 0.94 |  |
| L3 vs Spikformer (criterion 2) | Neo-mixer linear fusion, spikeDE LIF | Spikformer-style SSA (T=4) | -6.56 / -2.53 / -3.58 | -4.22 | worse | 0.25 | not met |
| L3 vs Spikformer (criterion 2) | Neo-mixer linear fusion, f-LIF | Spikformer-style SSA (T=4) | -5.62 / -4.60 / -4.83 | -5.02 | worse | 0.28 | not met |
| L3 vs Spikformer (criterion 2) | Neo-mixer linear fusion, population LIF (psi 0.5) | Spikformer-style SSA (T=4) | -4.14 / -0.92 / -4.67 | -3.24 | worse | 0.32 | not met |
| L3 vs Spikformer (criterion 2) | Neo-mixer linear fusion, population f-LIF (psi 1) | Spikformer-style SSA (T=4) | -6.24 / -1.90 / -2.27 | -3.47 | worse | 0.54 | not met |
