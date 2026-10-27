# Training budget `flif-v1-rhythm-budget-cinc` (CINC2017 fold 0) — summary

Best-validation weights on the test set; CinC score and macro-F1; mean ± sd over seeds (42, 7, 2026) (%). 30-epoch rows: stage 9 (`flif-v1-rhythm-cinc`) and the idea batch (`flif-v1-rhythm-bix-cinc`, population LIF). Schedule: timm cosine with warm-up 0.2 E, ending at 0 at epoch E.

| condition | epochs | n | score | macro-F1 | best val | best epoch | val − test | last-epoch train CE | total µJ (FFT) |
|---|---|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 30 | 3 | 65.60 ± 1.48 | 61.94 ± 0.41 | 68.79 ± 2.19 | 28.0 | 3.19 ± 0.76 | 0.245 ± 0.008 | 127.4 |
| Spikformer-style SSA (T=4) | 90 | 3 | 64.89 ± 0.80 | 60.26 ± 0.78 | 71.58 ± 1.12 | 57.7 | 6.68 ± 1.03 | 0.070 ± 0.004 | 130.9 |
| Neo-mixer OR, spikeDE LIF | 30 | 3 | 60.54 ± 0.56 | 56.07 ± 0.57 | 64.02 ± 1.65 | 23.0 | 3.48 ± 1.98 | 0.413 ± 0.001 | 28.0 |
| Neo-mixer OR, spikeDE LIF | 90 | 3 | 62.37 ± 0.88 | 56.31 ± 2.67 | 67.10 ± 1.03 | 61.7 | 4.73 ± 1.25 | 0.089 ± 0.002 | 29.2 |
| Neo-mixer OR, f-LIF | 30 | 3 | 59.75 ± 0.39 | 55.44 ± 1.24 | 63.66 ± 0.59 | 25.0 | 3.91 ± 0.94 | 0.408 ± 0.005 | 32.8 |
| Neo-mixer OR, f-LIF | 90 | 3 | 61.06 ± 2.05 | 55.52 ± 1.56 | 67.35 ± 0.70 | 62.7 | 6.30 ± 1.57 | 0.082 ± 0.002 | 34.4 |
| Neo-mixer OR, population LIF (psi 0.5) | 30 | 3 | 63.06 ± 0.83 | 57.22 ± 1.49 | 65.45 ± 0.92 | 26.0 | 2.39 ± 1.61 | 0.290 ± 0.011 | 36.5 |
| Neo-mixer OR, population LIF (psi 0.5) | 90 | 3 | 63.88 ± 1.08 | 59.85 ± 0.71 | 65.80 ± 1.07 | 73.0 | 1.92 ± 0.28 | 0.084 ± 0.003 | 36.4 |
| Neo-mixer OR, population f-LIF (psi 1) | 30 | 3 | 60.73 ± 1.67 | 55.74 ± 1.67 | 64.90 ± 0.85 | 25.7 | 4.17 ± 2.21 | 0.336 ± 0.002 | 65.2 |
| Neo-mixer OR, population f-LIF (psi 1) | 90 | 3 | 61.32 ± 0.55 | 56.70 ± 1.94 | 67.02 ± 1.29 | 80.3 | 5.70 ± 1.84 | 0.091 ± 0.003 | 65.3 |

## Pre-registered comparisons (%p, paired by seed; better = mean >= +0.5 and A > B in >= 2 of 3 seeds)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| E1 90 vs 30 epochs | Spikformer-style SSA (T=4) @90 | Spikformer-style SSA (T=4) @30 | -1.60 / +0.56 / -1.06 | -0.70 | worse | 1.03 |  |
| E1 90 vs 30 epochs | Neo-mixer OR, spikeDE LIF @90 | Neo-mixer OR, spikeDE LIF @30 | +1.15 / +2.58 / +1.77 | +1.83 | better | 1.04 |  |
| E1 90 vs 30 epochs | Neo-mixer OR, f-LIF @90 | Neo-mixer OR, f-LIF @30 | +2.41 / +2.51 / -0.99 | +1.31 | better | 1.05 |  |
| E1 90 vs 30 epochs | Neo-mixer OR, population LIF (psi 0.5) @90 | Neo-mixer OR, population LIF (psi 0.5) @30 | +1.22 / +1.36 / -0.13 | +0.82 | better | 1.00 |  |
| E1 90 vs 30 epochs | Neo-mixer OR, population f-LIF (psi 1) @90 | Neo-mixer OR, population f-LIF (psi 1) @30 | +1.95 / +0.93 / -1.12 | +0.59 | better | 1.00 |  |
| E3 vs Spikformer at 90 epochs (criterion 2) | Neo-mixer OR, spikeDE LIF @90 | Spikformer-style SSA (T=4) @90 | -4.29 / -2.18 / -1.10 | -2.52 | worse | 0.22 | not met |
| E3 vs Spikformer at 90 epochs (criterion 2) | Neo-mixer OR, f-LIF @90 | Spikformer-style SSA (T=4) @90 | -3.09 / -3.08 / -5.34 | -3.84 | worse | 0.26 | not met |
| E3 vs Spikformer at 90 epochs (criterion 2) | Neo-mixer OR, population LIF (psi 0.5) @90 | Spikformer-style SSA (T=4) @90 | -2.28 / +0.15 / -0.90 | -1.01 | worse | 0.28 | not met |
| E3 vs Spikformer at 90 epochs (criterion 2) | Neo-mixer OR, population f-LIF (psi 1) @90 | Spikformer-style SSA (T=4) @90 | -4.76 / -3.03 / -2.93 | -3.58 | worse | 0.50 | not met |

## E2 change of the gap to Spikformer-style (%p; g = gap@90 − gap@30, gap = Neo-mixer − Spikformer, same seed; primary: population LIF)

| Neo-mixer neuron | gap@30 per seed | gap@90 per seed | g per seed | mean g | verdict |
|---|---|---|---|---|---|
| spikeDE LIF | -7.05 / -4.19 / -3.93 | -4.29 / -2.18 / -1.10 | +2.76 / +2.01 / +2.83 | +2.53 | gap shrinks |
| f-LIF | -7.10 / -5.03 / -5.41 | -3.09 / -3.08 / -5.34 | +4.01 / +1.94 / +0.07 | +2.01 | gap shrinks |
| population LIF (psi 0.5) | -5.11 / -0.66 / -1.83 | -2.28 / +0.15 / -0.90 | +2.82 / +0.80 / +0.93 | +1.52 | gap shrinks |
| population f-LIF (psi 1) | -8.32 / -3.40 / -2.88 | -4.76 / -3.03 / -2.93 | +3.56 / +0.37 / -0.05 | +1.29 | gap shrinks |
