# Idea batch `flif-v1-rhythm-bix-cinc` (CINC2017) — summary

Best-validation weights on the test set; CinC score (mean F1 of N, AF, Other); mean ± sd over seeds (42, 7, 2026) (%). Stage-9 rows from `results/flif-v1-rhythm-cinc/raw.txt`. SHD rows compare structures within native-time processing only (no T = N claim on SHD).

| condition | n | score | best val | val − test | macro-F1 | total µJ (FFT / direct) | neocortex firing | params | time s |
|---|---|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 3 | 65.60 ± 1.48 | 68.79 ± 2.19 | 3.19 ± 0.76 | 61.94 ± 0.41 | 127.4 / 127.4 | n/a | 918088 | 400 |
| Spikformer-style SSA (T=4) + attention BN | 3 | 65.40 ± 0.57 | 71.06 ± 1.05 | 5.65 ± 0.54 | 61.18 ± 2.06 | 125.0 / 125.0 | n/a | 918856 | 398 |
| no mixing (floor) | 3 | 58.20 ± 1.94 | 59.74 ± 0.40 | 1.55 ± 2.16 | 54.28 ± 1.62 | 18.7 / 18.7 | n/a | 323328 | 210 |
| Neo-mixer (OR), spikeDE LIF | 3 | 60.54 ± 0.56 | 64.02 ± 1.65 | 3.48 ± 1.98 | 56.07 ± 0.57 | 28.0 / 28.0 | 0.0669 ± 0.0007 | 916224 | 3632 |
| Neo-mixer (OR), f-LIF | 3 | 59.75 ± 0.39 | 63.66 ± 0.59 | 3.91 ± 0.94 | 55.44 ± 1.24 | 32.8 / 85.2 | 0.0576 ± 0.0008 | 916224 | 7156 |
| Neo-mixer (OR), population LIF (psi 0.5) | 3 | 63.06 ± 0.83 | 65.45 ± 0.92 | 2.39 ± 1.61 | 57.22 ± 1.49 | 36.5 / 36.5 | 0.0974 ± 0.0020 | 922368 | 7830 |
| Neo-mixer (OR), population f-LIF (psi 1) | 3 | 60.73 ± 1.67 | 64.90 ± 0.85 | 4.17 ± 2.21 | 55.74 ± 1.67 | 65.2 / 266.7 | 0.0873 ± 0.0021 | 922368 | 7284 |
| Neo-mixer (OR), soma-less tau groups, LIF | 3 | 59.27 ± 0.45 | 61.30 ± 0.83 | 2.02 ± 0.74 | 55.01 ± 1.30 | 26.0 / 26.0 | 0.1023 ± 0.0021 | 916224 | 9554 |
| Neo-mixer (OR), soma-less tau groups, f-LIF | 3 | 59.51 ± 1.81 | 60.22 ± 1.40 | 0.70 ± 1.15 | 54.34 ± 0.37 | 30.5 / 82.8 | 0.0872 ± 0.0034 | 916224 | 14240 |
| Neo-mixer (OR), soma-less streams, LIF | 3 | 59.26 ± 1.43 | 59.10 ± 0.80 | -0.16 ± 0.77 | 54.68 ± 1.88 | 64.2 / 64.2 | 0.1044 ± 0.0007 | 936960 | 13736 |
| Neo-mixer (OR), soma-less streams, f-LIF | 3 | 58.30 ± 1.34 | 62.12 ± 1.60 | 3.83 ± 0.95 | 53.65 ± 1.72 | 81.2 / 290.6 | 0.0876 ± 0.0017 | 936960 | 24965 |
| BiX (head), spikeDE LIF | 3 | 61.50 ± 1.24 | 66.39 ± 1.09 | 4.89 ± 2.30 | 56.90 ± 0.90 | 35.8 / 35.8 | 0.0630 ± 0.0014 | 1511752 | 3860 |
| BiX (head), f-LIF | 3 | 61.11 ± 1.10 | 66.66 ± 1.67 | 5.55 ± 1.93 | 56.27 ± 1.55 | 40.1 / 92.5 | 0.0580 ± 0.0007 | 1511752 | 8485 |
| BiX (head), population LIF (psi 0.5) | 3 | 61.93 ± 0.02 | 64.54 ± 0.92 | 2.62 ± 0.91 | 56.82 ± 0.70 | 45.7 / 45.7 | 0.0556 ± 0.0039 | 1517896 | 8638 |
| BiX (head), population f-LIF (psi 1) | 3 | 59.62 ± 0.34 | 62.75 ± 0.78 | 3.13 ± 0.68 | 54.73 ± 2.23 | 73.6 / 275.1 | 0.0574 ± 0.0017 | 1517896 | 8271 |
| BiX (head), soma-less tau groups, LIF | 3 | 54.18 ± 1.90 | 56.02 ± 0.93 | 1.83 ± 2.41 | 50.41 ± 1.84 | 35.9 / 35.9 | 0.0814 ± 0.0023 | 1511752 | 11621 |
| BiX (head), soma-less tau groups, f-LIF | 3 | 54.53 ± 0.52 | 55.58 ± 0.64 | 1.04 ± 0.20 | 51.41 ± 0.19 | 40.8 / 93.1 | 0.0765 ± 0.0001 | 1511752 | 20292 |
| BiX (head), soma-less streams, LIF | 3 | 53.13 ± 2.01 | 55.25 ± 0.76 | 2.12 ± 1.45 | 48.90 ± 2.09 | 103.2 / 103.2 | 0.0846 ± 0.0007 | 1544008 | 10282 |
| BiX (head), soma-less streams, f-LIF | 3 | 54.68 ± 0.96 | 57.12 ± 0.60 | 2.44 ± 1.56 | 50.83 ± 1.12 | 122.3 / 331.7 | 0.0770 ± 0.0039 | 1544008 | 28030 |
| BiX-ch (per channel), spikeDE LIF | 3 | 60.36 ± 1.03 | 64.21 ± 2.81 | 3.85 ± 3.70 | 54.76 ± 1.33 | 36.1 / 36.1 | 0.0667 ± 0.0010 | 1509888 | 3790 |
| BiX-ch (per channel), f-LIF | 3 | 60.85 ± 1.26 | 64.77 ± 2.12 | 3.92 ± 2.31 | 55.31 ± 1.17 | 40.8 / 93.1 | 0.0609 ± 0.0013 | 1509888 | 8271 |
| BiX-ch (per channel), population LIF (psi 0.5) | 3 | 60.51 ± 1.22 | 64.27 ± 0.62 | 3.76 ± 0.96 | 54.33 ± 1.68 | 46.4 / 46.4 | 0.0638 ± 0.0025 | 1516032 | 8696 |
| BiX-ch (per channel), population f-LIF (psi 1) | 3 | 59.11 ± 0.50 | 62.79 ± 1.02 | 3.69 ± 1.23 | 52.87 ± 0.62 | 74.5 / 276.0 | 0.0665 ± 0.0008 | 1516032 | 8496 |

## Pre-registered comparisons (%p, paired by seed)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| B1 OR -> cross-attention | BiX (head), spikeDE LIF | Neo-mixer (OR), spikeDE LIF | +2.64 / +0.19 / +0.05 | +0.96 | better | 1.28 |  |
| B1 OR -> cross-attention | BiX (head), f-LIF | Neo-mixer (OR), f-LIF | +1.68 / +0.46 / +1.94 | +1.36 | better | 1.22 |  |
| B1 OR -> cross-attention | BiX (head), population LIF (psi 0.5) | Neo-mixer (OR), population LIF (psi 0.5) | -0.20 / -1.83 / -1.38 | -1.14 | worse | 1.25 |  |
| B1 OR -> cross-attention | BiX (head), population f-LIF (psi 1) | Neo-mixer (OR), population f-LIF (psi 1) | +1.05 / -1.70 / -2.68 | -1.11 | worse | 1.13 |  |
| B1 OR -> cross-attention | BiX (head), soma-less tau groups, LIF | Neo-mixer (OR), soma-less tau groups, LIF | -3.54 / -4.90 / -6.83 | -5.09 | worse | 1.38 |  |
| B1 OR -> cross-attention | BiX (head), soma-less tau groups, f-LIF | Neo-mixer (OR), soma-less tau groups, f-LIF | -2.45 / -5.49 / -7.00 | -4.98 | worse | 1.34 |  |
| B1 OR -> cross-attention | BiX (head), soma-less streams, LIF | Neo-mixer (OR), soma-less streams, LIF | -2.18 / -7.83 / -8.38 | -6.13 | worse | 1.61 |  |
| B1 OR -> cross-attention | BiX (head), soma-less streams, f-LIF | Neo-mixer (OR), soma-less streams, f-LIF | -3.40 / -5.05 / -2.41 | -3.62 | worse | 1.51 |  |
| B2 vs Spikformer (criterion 2) | BiX (head), spikeDE LIF | Spikformer-style SSA (T=4) | -4.40 / -4.00 / -3.87 | -4.09 | worse | 0.28 | not met |
| B2 vs Spikformer (criterion 2) | BiX (head), f-LIF | Spikformer-style SSA (T=4) | -5.42 / -4.57 / -3.47 | -4.49 | worse | 0.32 | not met |
| B2 vs Spikformer (criterion 2) | BiX (head), population LIF (psi 0.5) | Spikformer-style SSA (T=4) | -5.30 / -2.49 / -3.21 | -3.67 | worse | 0.36 | not met |
| B2 vs Spikformer (criterion 2) | BiX (head), population f-LIF (psi 1) | Spikformer-style SSA (T=4) | -7.27 / -5.11 / -5.55 | -5.98 | worse | 0.58 | not met |
| B2 vs Spikformer (criterion 2) | BiX (head), soma-less tau groups, LIF | Spikformer-style SSA (T=4) | -11.06 / -10.48 / -12.70 | -11.41 | worse | 0.28 | not met |
| B2 vs Spikformer (criterion 2) | BiX (head), soma-less tau groups, f-LIF | Spikformer-style SSA (T=4) | -12.20 / -9.88 / -11.11 | -11.06 | worse | 0.32 | not met |
| B2 vs Spikformer (criterion 2) | BiX (head), soma-less streams, LIF | Spikformer-style SSA (T=4) | -11.81 / -12.43 / -13.18 | -12.47 | worse | 0.81 | not met |
| B2 vs Spikformer (criterion 2) | BiX (head), soma-less streams, f-LIF | Spikformer-style SSA (T=4) | -13.55 / -9.71 / -9.50 | -10.92 | worse | 0.96 | not met |
| B2 vs Spikformer + attention BN (criterion 2) | BiX (head), spikeDE LIF | Spikformer-style SSA (T=4) + attention BN | -2.45 / -4.48 / -4.77 | -3.90 | worse | 0.29 | not met |
| B2 vs Spikformer + attention BN (criterion 2) | BiX (head), f-LIF | Spikformer-style SSA (T=4) + attention BN | -3.47 / -5.04 / -4.37 | -4.29 | worse | 0.32 | not met |
| B2 vs Spikformer + attention BN (criterion 2) | BiX (head), population LIF (psi 0.5) | Spikformer-style SSA (T=4) + attention BN | -3.36 / -2.97 / -4.11 | -3.48 | worse | 0.37 | not met |
| B2 vs Spikformer + attention BN (criterion 2) | BiX (head), population f-LIF (psi 1) | Spikformer-style SSA (T=4) + attention BN | -5.32 / -5.58 / -6.45 | -5.78 | worse | 0.59 | not met |
| B2 vs Spikformer + attention BN (criterion 2) | BiX (head), soma-less tau groups, LIF | Spikformer-style SSA (T=4) + attention BN | -9.11 / -10.96 / -13.60 | -11.22 | worse | 0.29 | not met |
| B2 vs Spikformer + attention BN (criterion 2) | BiX (head), soma-less tau groups, f-LIF | Spikformer-style SSA (T=4) + attention BN | -10.26 / -10.35 / -12.00 | -10.87 | worse | 0.33 | not met |
| B2 vs Spikformer + attention BN (criterion 2) | BiX (head), soma-less streams, LIF | Spikformer-style SSA (T=4) + attention BN | -9.86 / -12.90 / -14.07 | -12.28 | worse | 0.83 | not met |
| B2 vs Spikformer + attention BN (criterion 2) | BiX (head), soma-less streams, f-LIF | Spikformer-style SSA (T=4) + attention BN | -11.60 / -10.19 / -10.39 | -10.73 | worse | 0.98 | not met |
| B3 neuron inside BiX | BiX (head), f-LIF | BiX (head), spikeDE LIF | -1.02 / -0.56 / +0.40 | -0.39 | no clear difference | 1.12 |  |
| B3 neuron inside BiX | BiX (head), population LIF (psi 0.5) | BiX (head), spikeDE LIF | -0.90 / +1.51 / +0.66 | +0.42 | no clear difference | 1.28 |  |
| B3 neuron inside BiX | BiX (head), population f-LIF (psi 1) | BiX (head), spikeDE LIF | -2.87 / -1.10 / -1.68 | -1.88 | worse | 2.05 |  |
| B3 neuron inside BiX | BiX (head), soma-less tau groups, LIF | BiX (head), spikeDE LIF | -6.66 / -6.48 / -8.83 | -7.32 | worse | 1.00 |  |
| B3 neuron inside BiX | BiX (head), soma-less tau groups, f-LIF | BiX (head), spikeDE LIF | -7.80 / -5.88 / -7.23 | -6.97 | worse | 1.14 |  |
| B3 neuron inside BiX | BiX (head), soma-less streams, LIF | BiX (head), spikeDE LIF | -7.41 / -8.42 / -9.30 | -8.38 | worse | 2.88 |  |
| B3 neuron inside BiX | BiX (head), soma-less streams, f-LIF | BiX (head), spikeDE LIF | -9.15 / -5.71 / -5.62 | -6.83 | worse | 3.42 |  |
| B3 fractional order inside BiX | BiX (head), f-LIF | BiX (head), spikeDE LIF | -1.02 / -0.56 / +0.40 | -0.39 | no clear difference | 1.12 |  |
| B3 fractional order inside BiX | BiX (head), population f-LIF (psi 1) | BiX (head), population LIF (psi 0.5) | -1.96 / -2.61 / -2.34 | -2.31 | worse | 1.61 |  |
| B3 fractional order inside BiX | BiX (head), soma-less tau groups, f-LIF | BiX (head), soma-less tau groups, LIF | -1.14 / +0.60 / +1.59 | +0.35 | no clear difference | 1.14 |  |
| B3 fractional order inside BiX | BiX (head), soma-less streams, f-LIF | BiX (head), soma-less streams, LIF | -1.74 / +2.71 / +3.68 | +1.55 | better | 1.18 |  |
| B4 per-channel vs head form | BiX-ch (per channel), spikeDE LIF | BiX (head), spikeDE LIF | -1.71 / +0.33 / -2.05 | -1.14 | worse | 1.01 |  |
| B4 per-channel vs head form | BiX-ch (per channel), f-LIF | BiX (head), f-LIF | -0.33 / -0.45 / -0.00 | -0.26 | no clear difference | 1.02 |  |
| B4 per-channel vs head form | BiX-ch (per channel), population LIF (psi 0.5) | BiX (head), population LIF (psi 0.5) | -0.92 / -2.80 / -0.53 | -1.42 | worse | 1.01 |  |
| B4 per-channel vs head form | BiX-ch (per channel), population f-LIF (psi 1) | BiX (head), population f-LIF (psi 1) | -1.37 / +0.31 / -0.47 | -0.51 | worse | 1.01 |  |
| B4 per-channel vs Neo-mixer (OR) | BiX-ch (per channel), spikeDE LIF | Neo-mixer (OR), spikeDE LIF | +0.93 / +0.52 / -2.00 | -0.18 | no clear difference | 1.29 |  |
| B4 per-channel vs Neo-mixer (OR) | BiX-ch (per channel), f-LIF | Neo-mixer (OR), f-LIF | +1.35 / +0.01 / +1.93 | +1.10 | better | 1.24 |  |
| B4 per-channel vs Neo-mixer (OR) | BiX-ch (per channel), population LIF (psi 0.5) | Neo-mixer (OR), population LIF (psi 0.5) | -1.12 / -4.63 / -1.90 | -2.55 | worse | 1.27 |  |
| B4 per-channel vs Neo-mixer (OR) | BiX-ch (per channel), population f-LIF (psi 1) | Neo-mixer (OR), population f-LIF (psi 1) | -0.32 / -1.40 / -3.15 | -1.62 | worse | 1.14 |  |
| P1 soma -> tau groups (Neo-mixer) | Neo-mixer (OR), soma-less tau groups, LIF | Neo-mixer (OR), population LIF (psi 0.5) | -2.41 / -4.92 / -4.04 | -3.79 | worse | 0.71 |  |
| P1 soma -> streams (Neo-mixer) | Neo-mixer (OR), soma-less streams, LIF | Neo-mixer (OR), population LIF (psi 0.5) | -4.52 / -3.94 / -2.97 | -3.81 | worse | 1.76 |  |
| P1 groups -> streams (separation, Neo-mixer) | Neo-mixer (OR), soma-less streams, LIF | Neo-mixer (OR), soma-less tau groups, LIF | -2.11 / +0.99 / +1.07 | -0.02 | no clear difference | 2.47 |  |
| P1 soma -> tau groups (BiX) | BiX (head), soma-less tau groups, LIF | BiX (head), population LIF (psi 0.5) | -5.76 / -7.99 / -9.49 | -7.75 | worse | 0.79 |  |
| P1 soma -> streams (BiX) | BiX (head), soma-less streams, LIF | BiX (head), population LIF (psi 0.5) | -6.50 / -9.93 / -9.97 | -8.80 | worse | 2.26 |  |
| P1 groups -> streams (separation, BiX) | BiX (head), soma-less streams, LIF | BiX (head), soma-less tau groups, LIF | -0.75 / -1.94 / -0.48 | -1.05 | worse | 2.87 |  |
| P1 soma -> tau groups (Neo-mixer) | Neo-mixer (OR), soma-less tau groups, f-LIF | Neo-mixer (OR), population f-LIF (psi 1) | -1.43 / -0.98 / -1.23 | -1.22 | worse | 0.47 |  |
| P1 soma -> streams (Neo-mixer) | Neo-mixer (OR), soma-less streams, f-LIF | Neo-mixer (OR), population f-LIF (psi 1) | -1.83 / -1.26 / -4.21 | -2.43 | worse | 1.24 |  |
| P1 groups -> streams (separation, Neo-mixer) | Neo-mixer (OR), soma-less streams, f-LIF | Neo-mixer (OR), soma-less tau groups, f-LIF | -0.40 / -0.28 / -2.98 | -1.22 | worse | 2.66 |  |
| P1 soma -> tau groups (BiX) | BiX (head), soma-less tau groups, f-LIF | BiX (head), population f-LIF (psi 1) | -4.94 / -4.77 / -5.55 | -5.09 | worse | 0.55 |  |
| P1 soma -> streams (BiX) | BiX (head), soma-less streams, f-LIF | BiX (head), population f-LIF (psi 1) | -6.28 / -4.61 / -3.94 | -4.94 | worse | 1.66 |  |
| P1 groups -> streams (separation, BiX) | BiX (head), soma-less streams, f-LIF | BiX (head), soma-less tau groups, f-LIF | -1.34 / +0.17 / +1.61 | +0.14 | no clear difference | 3.00 |  |
| F1 fractional order, Neo-mixer population | Neo-mixer (OR), population f-LIF (psi 1) | Neo-mixer (OR), population LIF (psi 0.5) | -3.21 / -2.75 / -1.04 | -2.33 | worse | 1.78 |  |
| A1 attention BN | Spikformer-style SSA (T=4) + attention BN | Spikformer-style SSA (T=4) | -1.95 / +0.47 / +0.90 | -0.19 | no clear difference | 0.98 |  |
