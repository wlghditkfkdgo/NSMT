# Stage 9 `flif-v1-rhythm-cinc` (CINC2017) — summary

Best-validation weights on the test set; CinC score (mean F1 of N, AF, Other); mean ± sd over seeds (42, 7, 2026) (%).

| condition | n | score | macro-F1 (all classes) | accuracy | last-epoch score | synaptic µJ | total µJ (FFT / direct) | neocortex firing | params | time s |
|---|---|---|---|---|---|---|---|---|---|---|
| NSMT, spikeDE LIF | 3 | 61.76 ± 1.40 | 59.01 ± 0.95 | 70.01 ± 0.24 | 61.61 ± 0.72 | 2903.01 ± 54.83 | 3419.30 ± 57.63 / 3419.30 ± 57.63 | 0.0792 ± 0.0005 | 1262592 | 5144 |
| NSMT, f-LIF | 3 | 61.50 ± 1.20 | 58.59 ± 0.90 | 70.44 ± 0.28 | 60.53 ± 0.61 | 2956.84 ± 8.90 | 3480.00 ± 8.98 / 3519.27 ± 8.98 | 0.0881 ± 0.0001 | 1262592 | 7284 |
| NSMT, population f-LIF (calibrated) | 3 | 60.94 ± 2.24 | 55.05 ± 1.76 | 68.29 ± 1.27 | 60.57 ± 0.54 | 2716.47 ± 45.23 | 3247.40 ± 47.69 / 3398.52 ± 47.69 | 0.0398 ± 0.0022 | 1267200 | 7135 |
| NSMT, spikeDE LIF, plain CE | 3 | 59.22 ± 0.29 | 49.70 ± 1.31 | 69.05 ± 1.07 | 58.06 ± 0.68 | 2888.49 ± 55.48 | 3403.07 ± 58.32 / 3403.07 ± 58.32 | 0.0823 ± 0.0004 | 1262592 | 5233 |
| Hippo-only, Spikformer SSA (T=4) | 3 | 65.60 ± 1.48 | 61.94 ± 0.41 | 72.51 ± 0.72 | 65.36 ± 1.17 | 110.12 ± 0.68 | 127.36 ± 0.69 / 127.36 ± 0.69 | n/a | 918088 | 400 |
| Neo-mixer, bidirectional f-LIF | 3 | 59.75 ± 0.39 | 55.44 ± 1.24 | 69.64 ± 0.26 | 59.01 ± 0.38 | 24.18 ± 0.05 | 32.84 ± 0.05 / 85.19 ± 0.05 | 0.0576 ± 0.0008 | 916224 | 7156 |
| Neo-mixer, bidirectional population f-LIF | 3 | 60.73 ± 1.67 | 55.74 ± 1.67 | 68.29 ± 1.47 | 60.89 ± 1.26 | 22.46 ± 0.05 | 65.20 ± 0.05 / 266.71 ± 0.05 | 0.0873 ± 0.0021 | 922368 | 7284 |
| Neo-mixer, bidirectional spikeDE LIF | 3 | 60.54 ± 0.56 | 56.07 ± 0.57 | 69.42 ± 1.26 | 60.04 ± 0.56 | 24.28 ± 0.13 | 27.99 ± 0.13 / 27.99 ± 0.13 | 0.0669 ± 0.0007 | 916224 | 3632 |
| Neo-mixer, bidirectional LIF, plain CE | 3 | 58.68 ± 0.76 | 47.33 ± 0.72 | 69.25 ± 1.09 | 58.24 ± 1.23 | 24.55 ± 0.08 | 28.27 ± 0.08 / 28.27 ± 0.08 | 0.0695 ± 0.0012 | 916224 | 4329 |
| no token mixing (floor) | 3 | 58.20 ± 1.94 | 54.28 ± 1.62 | 68.60 ± 0.85 | 57.35 ± 1.19 | 17.07 ± 0.02 | 18.66 ± 0.02 / 18.66 ± 0.02 | n/a | 323328 | 210 |

## Pre-registered comparisons (%p, paired by seed)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| R1 memory requirement | NSMT, spikeDE LIF | no token mixing (floor) | +3.11 / +4.26 / +3.33 | +3.57 | better | 183.21 |  |
| R1 memory requirement | NSMT, f-LIF | no token mixing (floor) | +1.28 / +6.03 / +2.61 | +3.31 | better | 186.46 |  |
| R1 memory requirement | NSMT, population f-LIF (calibrated) | no token mixing (floor) | +4.16 / +5.25 / -1.18 | +2.74 | better | 173.99 |  |
| R1 memory requirement | Hippo-only, Spikformer SSA (T=4) | no token mixing (floor) | +8.41 / +8.40 / +5.39 | +7.40 | better | 6.82 |  |
| R1 memory requirement | Neo-mixer, bidirectional f-LIF | no token mixing (floor) | +1.31 / +3.37 / -0.02 | +1.56 | better | 1.76 |  |
| R1 memory requirement | Neo-mixer, bidirectional population f-LIF | no token mixing (floor) | +0.09 / +5.00 / +2.52 | +2.53 | better | 3.49 |  |
| R1 memory requirement | Neo-mixer, bidirectional spikeDE LIF | no token mixing (floor) | +1.36 / +4.21 / +1.47 | +2.35 | better | 1.50 |  |
| R2 neuron in NSMT | NSMT, f-LIF | NSMT, spikeDE LIF | -1.82 / +1.78 / -0.73 | -0.26 | no clear difference | 1.02 |  |
| R2 neuron in NSMT | NSMT, population f-LIF (calibrated) | NSMT, spikeDE LIF | +1.05 / +0.99 / -4.51 | -0.83 | no clear difference | 0.95 |  |
| R3 neuron in Neo-mixer | Neo-mixer, bidirectional f-LIF | Neo-mixer, bidirectional spikeDE LIF | -0.05 / -0.84 / -1.48 | -0.79 | worse | 1.17 |  |
| R3 neuron in Neo-mixer | Neo-mixer, bidirectional population f-LIF | Neo-mixer, bidirectional spikeDE LIF | -1.27 / +0.79 / +1.05 | +0.19 | no clear difference | 2.33 |  |
| R4 vs Hippo-only (Spikformer) | Neo-mixer, bidirectional f-LIF | Hippo-only, Spikformer SSA (T=4) | -7.10 / -5.03 / -5.41 | -5.84 | worse | 0.26 | not met |
| R4 vs Hippo-only (Spikformer) | Neo-mixer, bidirectional population f-LIF | Hippo-only, Spikformer SSA (T=4) | -8.32 / -3.40 / -2.88 | -4.87 | worse | 0.51 | not met |
| R4 vs Hippo-only (Spikformer) | Neo-mixer, bidirectional spikeDE LIF | Hippo-only, Spikformer SSA (T=4) | -7.05 / -4.19 / -3.93 | -5.05 | worse | 0.22 | not met |
| R4 vs Hippo-only (Spikformer) | NSMT, spikeDE LIF | Hippo-only, Spikformer SSA (T=4) | -5.30 / -4.14 / -2.06 | -3.83 | worse | 26.85 | not met |
| R4 vs Hippo-only (Spikformer) | NSMT, f-LIF | Hippo-only, Spikformer SSA (T=4) | -7.13 / -2.37 / -2.79 | -4.09 | worse | 27.32 | not met |
| R4 vs Hippo-only (Spikformer) | NSMT, population f-LIF (calibrated) | Hippo-only, Spikformer SSA (T=4) | -4.25 / -3.15 / -6.57 | -4.66 | worse | 25.50 | not met |
| R5 CAL vs plain CE (report) | NSMT, spikeDE LIF | NSMT, spikeDE LIF, plain CE | +3.07 / +0.86 / +3.70 | +2.54 | better | 1.00 |  |
| R5 CAL vs plain CE (report) | Neo-mixer, bidirectional spikeDE LIF | Neo-mixer, bidirectional LIF, plain CE | +0.95 / +1.27 / +3.38 | +1.87 | better | 0.99 |  |
