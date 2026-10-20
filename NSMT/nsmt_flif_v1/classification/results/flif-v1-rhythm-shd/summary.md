# Stage 9 `flif-v1-rhythm-shd` (SHD) — summary

Best-validation weights on the test set; test accuracy; mean ± sd over seeds (42, 7, 2026, 123, 1024) (%).

| condition | n | score | macro-F1 (all classes) | accuracy | last-epoch score | synaptic µJ | total µJ (FFT / direct) | neocortex firing | params | time s |
|---|---|---|---|---|---|---|---|---|---|---|
| NSMT, spikeDE LIF | 5 | 80.37 ± 1.05 | 80.13 ± 1.10 | 80.37 ± 1.05 | 80.76 ± 0.57 | 26113.28 ± 9.54 | 26484.59 ± 9.99 / 26484.59 ± 9.99 | 0.0825 ± 0.0010 | 2807808 | 7516 |
| NSMT, f-LIF | 5 | 81.34 ± 0.61 | 81.16 ± 0.62 | 81.34 ± 0.61 | 80.87 ± 0.38 | 26109.73 ± 7.39 | 26484.08 ± 7.85 / 26512.40 ± 7.85 | 0.0761 ± 0.0018 | 2807808 | 11027 |
| NSMT, population f-LIF (calibrated) | 5 | 86.32 ± 0.92 | 85.46 ± 1.04 | 86.32 ± 0.92 | 86.34 ± 0.73 | 26105.50 ± 11.70 | 26508.12 ± 12.97 / 26616.32 ± 12.97 | 0.1197 ± 0.0076 | 2812416 | 11047 |
| Hippo-only, Spikformer SSA (T=4) | 5 | 86.75 ± 1.32 | 86.41 ± 1.35 | 86.75 ± 1.32 | 86.77 ± 0.97 | 1056.14 ± 0.45 | 1071.24 ± 0.46 / 1071.24 ± 0.46 | n/a | 1436984 | 902 |
| Neo-mixer, bidirectional f-LIF | 5 | 78.71 ± 1.47 | 78.55 ± 1.45 | 78.71 ± 1.47 | 79.07 ± 0.31 | 260.38 ± 0.14 | 267.65 ± 0.14 / 305.41 ± 0.14 | 0.0695 ± 0.0015 | 1435392 | 10747 |
| Neo-mixer, bidirectional population f-LIF | 5 | 86.21 ± 0.68 | 85.75 ± 0.66 | 86.21 ± 0.68 | 86.44 ± 0.57 | 258.72 ± 0.20 | 294.58 ± 0.20 / 438.85 ± 0.20 | 0.1243 ± 0.0014 | 1441536 | 11395 |
| Neo-mixer, bidirectional spikeDE LIF | 5 | 80.39 ± 0.85 | 80.29 ± 0.86 | 80.39 ± 0.85 | 80.29 ± 0.50 | 260.38 ± 0.05 | 263.59 ± 0.05 / 263.59 ± 0.05 | 0.0740 ± 0.0008 | 1435392 | 6039 |
| no token mixing (floor) | 5 | 76.79 ± 1.07 | 76.72 ± 1.14 | 76.79 ± 1.07 | 77.09 ± 0.90 | 251.61 ± 0.04 | 253.01 ± 0.04 / 253.01 ± 0.04 | n/a | 842496 | 630 |

## Pre-registered comparisons (%p, paired by seed)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| R1 memory requirement | NSMT, spikeDE LIF | no token mixing (floor) | +5.43 / +2.87 / +2.21 / +3.40 / +3.98 | +3.58 | better | 104.68 |  |
| R1 memory requirement | NSMT, f-LIF | no token mixing (floor) | +5.70 / +4.37 / +2.74 / +3.53 / +6.40 | +4.55 | better | 104.68 |  |
| R1 memory requirement | NSMT, population f-LIF (calibrated) | no token mixing (floor) | +8.92 / +8.44 / +9.67 / +8.57 / +12.01 | +9.52 | better | 104.77 |  |
| R1 memory requirement | Hippo-only, Spikformer SSA (T=4) | no token mixing (floor) | +11.44 / +9.45 / +7.60 / +10.73 / +10.56 | +9.96 | better | 4.23 |  |
| R1 memory requirement | Neo-mixer, bidirectional f-LIF | no token mixing (floor) | +0.04 / +2.65 / +1.19 / +2.52 / +3.18 | +1.92 | better | 1.06 |  |
| R1 memory requirement | Neo-mixer, bidirectional population f-LIF | no token mixing (floor) | +10.73 / +8.48 / +8.66 / +7.82 / +11.40 | +9.42 | better | 1.16 |  |
| R1 memory requirement | Neo-mixer, bidirectional spikeDE LIF | no token mixing (floor) | +4.64 / +3.71 / +2.16 / +1.55 / +5.92 | +3.60 | better | 1.04 |  |
| R2 neuron in NSMT | NSMT, f-LIF | NSMT, spikeDE LIF | +0.27 / +1.50 / +0.53 / +0.13 / +2.43 | +0.97 | better | 1.00 |  |
| R2 neuron in NSMT | NSMT, population f-LIF (calibrated) | NSMT, spikeDE LIF | +3.49 / +5.57 / +7.46 / +5.17 / +8.04 | +5.95 | better | 1.00 |  |
| R3 neuron in Neo-mixer | Neo-mixer, bidirectional f-LIF | Neo-mixer, bidirectional spikeDE LIF | -4.59 / -1.06 / -0.97 / +0.97 / -2.74 | -1.68 | worse | 1.02 |  |
| R3 neuron in Neo-mixer | Neo-mixer, bidirectional population f-LIF | Neo-mixer, bidirectional spikeDE LIF | +6.10 / +4.77 / +6.49 / +6.27 / +5.48 | +5.82 | better | 1.12 |  |
| R4 vs Hippo-only (Spikformer) | Neo-mixer, bidirectional f-LIF | Hippo-only, Spikformer SSA (T=4) | -11.40 / -6.80 / -6.40 / -8.22 / -7.38 | -8.04 | worse | 0.25 | not met |
| R4 vs Hippo-only (Spikformer) | Neo-mixer, bidirectional population f-LIF | Hippo-only, Spikformer SSA (T=4) | -0.71 / -0.97 / +1.06 / -2.92 / +0.84 | -0.54 | no clear difference | 0.27 | not met |
| R4 vs Hippo-only (Spikformer) | Neo-mixer, bidirectional spikeDE LIF | Hippo-only, Spikformer SSA (T=4) | -6.80 / -5.74 / -5.43 / -9.19 / -4.64 | -6.36 | worse | 0.25 | not met |
| R4 vs Hippo-only (Spikformer) | NSMT, spikeDE LIF | Hippo-only, Spikformer SSA (T=4) | -6.01 / -6.58 / -5.39 / -7.33 / -6.58 | -6.38 | worse | 24.72 | not met |
| R4 vs Hippo-only (Spikformer) | NSMT, f-LIF | Hippo-only, Spikformer SSA (T=4) | -5.74 / -5.08 / -4.86 / -7.20 / -4.15 | -5.41 | worse | 24.72 | not met |
| R4 vs Hippo-only (Spikformer) | NSMT, population f-LIF (calibrated) | Hippo-only, Spikformer SSA (T=4) | -2.52 / -1.02 / +2.08 / -2.16 / +1.46 | -0.43 | no clear difference | 24.75 | not met |
