# Spikformer + T = N neocortex added, 90 epochs `flif-v1-rhythm-addneo90-cinc` (CINC2017 fold 0) — summary

Best-validation weights on the test set; CinC score and macro-F1; mean ± sd over seeds (42, 7, 2026) (%). 30-epoch rows: `flif-v1-rhythm-addneo-cinc` and stage 9 (`flif-v1-rhythm-cinc`, Spikformer-style); Spikformer-style @90: `flif-v1-rhythm-budget-cinc`.

| condition | epochs | n | score | macro-F1 | best val | best epoch | val − test | last-epoch train CE | total µJ (FFT) |
|---|---|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 30 | 3 | 65.60 ± 1.48 | 61.94 ± 0.41 | 68.79 ± 2.19 | 28.0 | 3.19 ± 0.76 | 0.245 ± 0.008 | 127.4 |
| Spikformer-style SSA (T=4) | 90 | 3 | 64.89 ± 0.80 | 60.26 ± 0.78 | 71.58 ± 1.12 | 57.7 | 6.68 ± 1.03 | 0.070 ± 0.004 | 130.9 |
| Spikformer + neocortex added, LIF, memoryless (reset every token) | 30 | 3 | 65.84 ± 1.92 | 59.43 ± 0.81 | 69.21 ± 0.76 | 25.3 | 3.38 ± 2.55 | 0.196 ± 0.003 | 161.6 |
| Spikformer + neocortex added, LIF, memoryless (reset every token) | 90 | 3 | 66.04 ± 0.27 | 61.48 ± 0.12 | 71.06 ± 1.30 | 60.7 | 5.02 ± 1.46 | 0.059 ± 0.002 | 168.0 |
| Spikformer + T = N neocortex added, spikeDE LIF | 30 | 3 | 63.34 ± 1.55 | 57.32 ± 0.99 | 67.38 ± 1.18 | 20.3 | 4.04 ± 1.53 | 0.205 ± 0.009 | 165.4 |
| Spikformer + T = N neocortex added, spikeDE LIF | 90 | 3 | 65.49 ± 0.44 | 60.90 ± 0.98 | 70.03 ± 0.79 | 49.3 | 4.54 ± 1.02 | 0.058 ± 0.005 | 170.9 |
| Spikformer + T = N neocortex added, f-LIF | 30 | 3 | 64.35 ± 0.43 | 59.89 ± 0.90 | 68.24 ± 0.20 | 22.3 | 3.89 ± 0.24 | 0.193 ± 0.001 | 170.1 |
| Spikformer + T = N neocortex added, f-LIF | 90 | 3 | 64.25 ± 1.48 | 59.42 ± 1.93 | 70.58 ± 1.47 | 53.3 | 6.33 ± 2.50 | 0.056 ± 0.003 | 174.1 |
| Spikformer + T = N neocortex added, population LIF (psi 0.5) | 30 | 3 | 66.06 ± 0.71 | 59.62 ± 2.28 | 70.21 ± 1.01 | 23.0 | 4.14 ± 0.40 | 0.224 ± 0.014 | 172.8 |
| Spikformer + T = N neocortex added, population LIF (psi 0.5) | 90 | 3 | 65.38 ± 1.24 | 60.23 ± 2.23 | 71.98 ± 0.59 | 48.0 | 6.60 ± 1.00 | 0.060 ± 0.001 | 179.6 |
| Spikformer + T = N neocortex added, population f-LIF (psi 1) | 30 | 3 | 64.72 ± 0.40 | 59.15 ± 0.98 | 69.89 ± 1.13 | 24.7 | 5.17 ± 0.82 | 0.224 ± 0.004 | 191.5 |
| Spikformer + T = N neocortex added, population f-LIF (psi 1) | 90 | 3 | 65.28 ± 1.65 | 60.47 ± 1.27 | 71.95 ± 0.17 | 59.7 | 6.67 ± 1.81 | 0.056 ± 0.002 | 196.9 |

## Pre-registered comparisons (%p, paired by seed; better = mean >= +0.5 and A > B in >= 2 of 3 seeds)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| D1 vs Spikformer at 90 epochs (criterion 2) | Spikformer + neocortex added, LIF, memoryless (reset every token) @90 | Spikformer-style SSA (T=4) @90 | +0.09 / +1.28 / +2.06 | +1.14 | better | 1.28 | met |
| D1 vs Spikformer at 90 epochs (criterion 2) | Spikformer + T = N neocortex added, spikeDE LIF @90 | Spikformer-style SSA (T=4) @90 | +0.24 / +0.60 / +0.96 | +0.60 | better | 1.31 | met |
| D1 vs Spikformer at 90 epochs (criterion 2) | Spikformer + T = N neocortex added, f-LIF @90 | Spikformer-style SSA (T=4) @90 | +0.01 / -2.27 / +0.32 | -0.64 | no clear difference | 1.33 | not met |
| D1 vs Spikformer at 90 epochs (criterion 2) | Spikformer + T = N neocortex added, population LIF (psi 0.5) @90 | Spikformer-style SSA (T=4) @90 | +0.36 / +1.19 / -0.11 | +0.48 | no clear difference | 1.37 | not met |
| D1 vs Spikformer at 90 epochs (criterion 2) | Spikformer + T = N neocortex added, population f-LIF (psi 1) @90 | Spikformer-style SSA (T=4) @90 | -1.83 / -0.03 / +3.02 | +0.39 | no clear difference | 1.50 | not met |
| D2 T = N memory vs memoryless neocortex at 90 epochs | Spikformer + T = N neocortex added, spikeDE LIF @90 | Spikformer + neocortex added, LIF, memoryless (reset every token) @90 | +0.15 / -0.68 / -1.10 | -0.54 | worse | 1.02 |  |
| D2 T = N memory vs memoryless neocortex at 90 epochs | Spikformer + T = N neocortex added, f-LIF @90 | Spikformer + neocortex added, LIF, memoryless (reset every token) @90 | -0.07 / -3.55 / -1.73 | -1.79 | worse | 1.04 |  |
| D2 T = N memory vs memoryless neocortex at 90 epochs | Spikformer + T = N neocortex added, population LIF (psi 0.5) @90 | Spikformer + neocortex added, LIF, memoryless (reset every token) @90 | +0.28 / -0.09 / -2.17 | -0.66 | worse | 1.07 |  |
| D2 T = N memory vs memoryless neocortex at 90 epochs | Spikformer + T = N neocortex added, population f-LIF (psi 1) @90 | Spikformer + neocortex added, LIF, memoryless (reset every token) @90 | -1.91 / -1.31 / +0.96 | -0.75 | worse | 1.17 |  |
| D3 90 vs 30 epochs | Spikformer + neocortex added, LIF, memoryless (reset every token) @90 | Spikformer + neocortex added, LIF, memoryless (reset every token) @30 | -2.12 / +0.64 / +2.07 | +0.20 | no clear difference | 1.04 |  |
| D3 90 vs 30 epochs | Spikformer + T = N neocortex added, spikeDE LIF @90 | Spikformer + T = N neocortex added, spikeDE LIF @30 | +2.38 / +0.78 / +3.31 | +2.16 | better | 1.03 |  |
| D3 90 vs 30 epochs | Spikformer + T = N neocortex added, f-LIF @90 | Spikformer + T = N neocortex added, f-LIF @30 | +0.99 / -1.16 / -0.14 | -0.10 | no clear difference | 1.02 |  |
| D3 90 vs 30 epochs | Spikformer + T = N neocortex added, population LIF (psi 0.5) @90 | Spikformer + T = N neocortex added, population LIF (psi 0.5) @30 | -0.80 / +0.20 / -1.46 | -0.69 | worse | 1.04 |  |
| D3 90 vs 30 epochs | Spikformer + T = N neocortex added, population f-LIF (psi 1) @90 | Spikformer + T = N neocortex added, population f-LIF (psi 1) @30 | -1.03 / -0.09 / +2.80 | +0.56 | no clear difference | 1.03 |  |
| D4 (report) @90 vs Spikformer @30 | Spikformer + neocortex added, LIF, memoryless (reset every token) @90 | Spikformer-style SSA (T=4) @30 | -1.52 / +1.84 / +1.00 | +0.44 | no clear difference | 1.32 |  |
| D4 (report) @90 vs Spikformer @30 | Spikformer + T = N neocortex added, spikeDE LIF @90 | Spikformer-style SSA (T=4) @30 | -1.37 / +1.16 / -0.11 | -0.10 | no clear difference | 1.34 |  |
| D4 (report) @90 vs Spikformer @30 | Spikformer + T = N neocortex added, f-LIF @90 | Spikformer-style SSA (T=4) @30 | -1.59 / -1.71 / -0.74 | -1.35 | worse | 1.37 |  |
| D4 (report) @90 vs Spikformer @30 | Spikformer + T = N neocortex added, population LIF (psi 0.5) @90 | Spikformer-style SSA (T=4) @30 | -1.24 / +1.75 / -1.17 | -0.22 | no clear difference | 1.41 |  |
| D4 (report) @90 vs Spikformer @30 | Spikformer + T = N neocortex added, population f-LIF (psi 1) @90 | Spikformer-style SSA (T=4) @30 | -3.43 / +0.53 / +1.96 | -0.31 | no clear difference | 1.55 |  |
