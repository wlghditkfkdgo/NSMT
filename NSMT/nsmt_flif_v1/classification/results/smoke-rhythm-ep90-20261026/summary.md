# Training budget `smoke-rhythm-ep90-20261026` (CINC2017 fold 0) — summary

Best-validation weights on the test set; CinC score and macro-F1; mean ± sd over seeds (42, 7, 2026) (%). 30-epoch rows: stage 9 (`flif-v1-rhythm-cinc`) and the idea batch (`flif-v1-rhythm-bix-cinc`, population LIF). Schedule: timm cosine with warm-up 0.2 E, ending at 0 at epoch E.

| condition | epochs | n | score | macro-F1 | best val | best epoch | val − test | last-epoch train CE | total µJ (FFT) |
|---|---|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 30 | 3 | 65.60 ± 1.48 | 61.94 ± 0.41 | 68.79 ± 2.19 | 28.0 | 3.19 ± 0.76 | 0.245 ± 0.008 | 127.4 |
| Spikformer-style SSA (T=4) | 90 | 1 | 37.88 ± 0.00 | 28.41 ± 0.00 | 37.39 ± 0.00 | 0.0 | -0.49 ± 0.00 | 1.120 ± 0.000 | 107.6 |
| Neo-mixer OR, spikeDE LIF | 30 | 3 | 60.54 ± 0.56 | 56.07 ± 0.57 | 64.02 ± 1.65 | 23.0 | 3.48 ± 1.98 | 0.413 ± 0.001 | 28.0 |
| Neo-mixer OR, spikeDE LIF | 90 | 1 | 28.36 ± 0.00 | 21.27 ± 0.00 | 26.99 ± 0.00 | 0.0 | -1.37 ± 0.00 | 1.150 ± 0.000 | 24.4 |
| Neo-mixer OR, f-LIF | 30 | 3 | 59.75 ± 0.39 | 55.44 ± 1.24 | 63.66 ± 0.59 | 25.0 | 3.91 ± 0.94 | 0.408 ± 0.005 | 32.8 |
| Neo-mixer OR, f-LIF | 90 | 1 | 28.99 ± 0.00 | 21.74 ± 0.00 | 27.34 ± 0.00 | 0.0 | -1.65 ± 0.00 | 1.150 ± 0.000 | 29.5 |
| Neo-mixer OR, population LIF (psi 0.5) | 30 | 3 | 63.06 ± 0.83 | 57.22 ± 1.49 | 65.45 ± 0.92 | 26.0 | 2.39 ± 1.61 | 0.290 ± 0.011 | 36.5 |
| Neo-mixer OR, population LIF (psi 0.5) | 90 | 1 | 35.25 ± 0.00 | 26.44 ± 0.00 | 34.24 ± 0.00 | 0.0 | -1.00 ± 0.00 | 1.139 ± 0.000 | 35.2 |
| Neo-mixer OR, population f-LIF (psi 1) | 30 | 3 | 60.73 ± 1.67 | 55.74 ± 1.67 | 64.90 ± 0.85 | 25.7 | 4.17 ± 2.21 | 0.336 ± 0.002 | 65.2 |
| Neo-mixer OR, population f-LIF (psi 1) | 90 | 1 | 32.52 ± 0.00 | 24.39 ± 0.00 | 31.68 ± 0.00 | 0.0 | -0.83 ± 0.00 | 1.141 ± 0.000 | 63.1 |

## Pre-registered comparisons (%p, paired by seed; better = mean >= +0.5 and A > B in >= 2 of 3 seeds)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| E1 90 vs 30 epochs | Spikformer-style SSA (T=4) @90 | Spikformer-style SSA (T=4) @30 |  | +nan | incomplete | 0.85 |  |
| E1 90 vs 30 epochs | Neo-mixer OR, spikeDE LIF @90 | Neo-mixer OR, spikeDE LIF @30 |  | +nan | incomplete | 0.87 |  |
| E1 90 vs 30 epochs | Neo-mixer OR, f-LIF @90 | Neo-mixer OR, f-LIF @30 |  | +nan | incomplete | 0.90 |  |
| E1 90 vs 30 epochs | Neo-mixer OR, population LIF (psi 0.5) @90 | Neo-mixer OR, population LIF (psi 0.5) @30 |  | +nan | incomplete | 0.96 |  |
| E1 90 vs 30 epochs | Neo-mixer OR, population f-LIF (psi 1) @90 | Neo-mixer OR, population f-LIF (psi 1) @30 |  | +nan | incomplete | 0.97 |  |
| E3 vs Spikformer at 90 epochs (criterion 2) | Neo-mixer OR, spikeDE LIF @90 | Spikformer-style SSA (T=4) @90 |  | +nan | incomplete | 0.23 |  |
| E3 vs Spikformer at 90 epochs (criterion 2) | Neo-mixer OR, f-LIF @90 | Spikformer-style SSA (T=4) @90 |  | +nan | incomplete | 0.27 |  |
| E3 vs Spikformer at 90 epochs (criterion 2) | Neo-mixer OR, population LIF (psi 0.5) @90 | Spikformer-style SSA (T=4) @90 |  | +nan | incomplete | 0.33 |  |
| E3 vs Spikformer at 90 epochs (criterion 2) | Neo-mixer OR, population f-LIF (psi 1) @90 | Spikformer-style SSA (T=4) @90 |  | +nan | incomplete | 0.59 |  |

## E2 change of the gap to Spikformer-style (%p; g = gap@90 − gap@30, gap = Neo-mixer − Spikformer, same seed; primary: population LIF)

| Neo-mixer neuron | gap@30 per seed | gap@90 per seed | g per seed | mean g | verdict |
|---|---|---|---|---|---|
| spikeDE LIF | | | | | incomplete |
| f-LIF | | | | | incomplete |
| population LIF (psi 0.5) | | | | | incomplete |
| population f-LIF (psi 1) | | | | | incomplete |
