# Cause-separation `flif-v1-rhythm-fix-cinc` (CINC2017) — summary

Best-validation weights on the test set; CinC score (mean F1 of N, AF, Other); mean ± sd over seeds (42, 7, 2026) (%). Stage-9 rows come from `results/flif-v1-rhythm-cinc/raw.txt`.

| condition | n | score | best val | val − test | macro-F1 | last-epoch score | total µJ (FFT) | neocortex / mixer firing | params | time s |
|---|---|---|---|---|---|---|---|---|---|---|
| Hippo-only, Spikformer SSA (T=4), relative bias | 3 | 65.60 ± 1.48 | 68.79 ± 2.19 | 3.19 ± 0.76 | 61.94 ± 0.41 | 65.36 ± 1.17 | 127.4 | n/a | 918088 | 400 |
| Hippo-only, Spikformer SSA (T=4), no position | 3 | 64.48 ± 0.94 | 66.97 ± 0.94 | 2.49 ± 0.80 | 58.32 ± 1.70 | 64.18 ± 0.48 | 125.0 | n/a | 916224 | 460 |
| NSMT LIF (T=N) | 3 | 61.76 ± 1.40 | 66.00 ± 1.03 | 4.23 ± 0.57 | 59.01 ± 0.95 | 61.61 ± 0.72 | 3419.3 | 0.0792 ± 0.0005 | 1262592 | 5144 |
| NSMT LIF + absolute position | 3 | 60.42 ± 1.04 | 66.73 ± 1.10 | 6.31 ± 2.04 | 56.84 ± 1.11 | 61.07 ± 0.14 | 3438.9 | 0.0815 ± 0.0009 | 1307520 | 4264 |
| NSMT LIF, hippocampus T=4 + token cross-attention | 3 | 63.42 ± 0.85 | 68.37 ± 0.97 | 4.95 ± 0.30 | 55.80 ± 0.88 | 63.56 ± 0.75 | 148.1 | 0.0590 ± 0.0017 | 1264456 | 2931 |
| NSMT population (T=N, psi 1) | 3 | 60.94 ± 2.24 | 65.88 ± 0.60 | 4.94 ± 1.64 | 55.05 ± 1.76 | 60.57 ± 0.54 | 3247.4 | 0.0398 ± 0.0022 | 1267200 | 7135 |
| NSMT population + absolute position | 3 | 61.67 ± 1.43 | 66.00 ± 0.36 | 4.32 ± 1.07 | 55.97 ± 2.08 | 61.11 ± 0.91 | 3296.9 | 0.0437 ± 0.0011 | 1312128 | 5659 |
| NSMT population, hippocampus T=4 + token cross-attention | 3 | 58.66 ± 2.25 | 64.12 ± 1.83 | 5.46 ± 0.43 | 52.12 ± 1.11 | 59.36 ± 1.92 | 160.0 | 0.0314 ± 0.0004 | 1269064 | 5018 |
| NSMT population psi 1.5 | 3 | 61.42 ± 0.84 | 66.16 ± 0.41 | 4.73 ± 1.19 | 58.55 ± 0.24 | 60.09 ± 1.41 | 3386.0 | 0.0658 ± 0.0044 | 1267200 | 5596 |
| NSMT population psi 2 | 3 | 62.10 ± 1.62 | 66.86 ± 1.71 | 4.76 ± 1.48 | 57.32 ± 0.03 | 61.44 ± 1.23 | 3397.6 | 0.1023 ± 0.0047 | 1267200 | 5554 |
| Neo-mixer bidir spikeDE LIF | 3 | 60.54 ± 0.56 | 64.02 ± 1.65 | 3.48 ± 1.98 | 56.07 ± 0.57 | 60.04 ± 0.56 | 28.0 | 0.0669 ± 0.0007 | 916224 | 3632 |
| Neo-mixer bidir LIF + absolute position | 3 | 58.58 ± 0.83 | 64.00 ± 1.02 | 5.42 ± 1.66 | 54.47 ± 1.16 | 58.73 ± 0.50 | 28.2 | 0.0685 ± 0.0015 | 961152 | 4030 |
| Neo-mixer bidir spikingjelly LIF (control) | 3 | 59.28 ± 0.37 | 64.95 ± 0.88 | 5.67 ± 0.66 | 54.08 ± 0.90 | 59.01 ± 0.36 | 27.6 | 0.0450 ± 0.0004 | 916224 | 1476 |
| Neo-mixer bidir recurrent spikingjelly LIF | 3 | 45.99 ± 9.22 | 44.18 ± 9.58 | -1.81 ± 0.72 | 37.19 ± 9.48 | 41.53 ± 14.42 | 30.2 | 0.0837 ± 0.0159 | 1506052 | 2278 |
| Neo-mixer bidir population (taus 4-32, psi 1) | 3 | 60.73 ± 1.67 | 64.90 ± 0.85 | 4.17 ± 2.21 | 55.74 ± 1.67 | 60.89 ± 1.26 | 65.2 | 0.0873 ± 0.0021 | 922368 | 7284 |
| Neo-mixer bidir population + absolute position | 3 | 58.78 ± 1.04 | 62.81 ± 1.64 | 4.03 ± 2.58 | 52.96 ± 1.48 | 58.38 ± 0.54 | 65.3 | 0.0939 ± 0.0018 | 967296 | 5535 |
| Neo-mixer bidir recurrent population f-LIF | 3 | 50.19 ± 2.30 | 49.34 ± 3.33 | -0.86 ± 1.04 | 46.21 ± 2.10 | 49.76 ± 2.96 | 68.9 | 0.0923 ± 0.0098 | 1512196 | 5022 |
| Neo-mixer bidir population (taus 16-128) | 3 | 60.10 ± 0.23 | 64.85 ± 1.80 | 4.75 ± 2.02 | 55.33 ± 1.18 | 59.73 ± 1.12 | 64.2 | 0.0602 ± 0.0014 | 922368 | 7928 |
| Neo-mixer bidir population psi 1.5 | 3 | 60.97 ± 0.53 | 63.92 ± 1.28 | 2.94 ± 1.44 | 56.20 ± 1.33 | 60.21 ± 0.86 | 65.9 | 0.1190 ± 0.0021 | 922368 | 5209 |
| Neo-mixer bidir population psi 2 | 3 | 60.81 ± 0.74 | 64.28 ± 1.80 | 3.47 ± 2.39 | 56.72 ± 1.14 | 60.82 ± 0.44 | 66.7 | 0.1436 ± 0.0016 | 922368 | 4704 |

## Pre-registered comparisons (%p, paired by seed)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| Y1 relative bias removed | Hippo-only, Spikformer SSA (T=4), no position | Hippo-only, Spikformer SSA (T=4), relative bias | -1.74 / -0.17 / -1.44 | -1.11 | worse | 0.98 |  |
| Y2 absolute position in NSMT | NSMT LIF + absolute position | NSMT LIF (T=N) | -2.42 / +1.29 / -2.88 | -1.34 | worse | 1.01 |  |
| Y3 absolute position in Neo-mixer | Neo-mixer bidir LIF + absolute position | Neo-mixer bidir spikeDE LIF | -0.68 / -2.02 / -3.19 | -1.97 | worse | 1.01 |  |
| Y4 position matched (none): NSMT+pos vs SSA no-pos | NSMT LIF + absolute position | Hippo-only, Spikformer SSA (T=4), no position | -5.99 / -2.69 / -3.50 | -4.06 | worse | 27.51 |  |
| Y4 position matched (none): mixer+pos vs SSA no-pos | Neo-mixer bidir LIF + absolute position | Hippo-only, Spikformer SSA (T=4), no position | -5.99 / -6.04 / -5.68 | -5.90 | worse | 0.23 |  |
| Y4 vs Hippo-only (criterion 2) | NSMT LIF + absolute position | Hippo-only, Spikformer SSA (T=4), relative bias | -7.73 / -2.86 / -4.94 | -5.17 | worse | 27.00 | not met |
| Y4 vs Hippo-only (criterion 2) | Neo-mixer bidir LIF + absolute position | Hippo-only, Spikformer SSA (T=4), relative bias | -7.73 / -6.21 / -7.12 | -7.02 | worse | 0.22 | not met |
| Z1 recurrence (same neuron) | Neo-mixer bidir recurrent spikingjelly LIF | Neo-mixer bidir spikingjelly LIF (control) | -6.88 / -8.82 / -24.17 | -13.29 | worse | 1.10 |  |
| Z1 report: spikingjelly vs spikeDE LIF | Neo-mixer bidir spikingjelly LIF (control) | Neo-mixer bidir spikeDE LIF | -1.35 / -0.79 / -1.65 | -1.26 | worse | 0.98 |  |
| Z2 slower population branches | Neo-mixer bidir population (taus 16-128) | Neo-mixer bidir population (taus 4-32, psi 1) | +1.10 / -0.66 / -2.33 | -0.63 | worse | 0.98 |  |
| Z3 vs Hippo-only (criterion 2) | Neo-mixer bidir recurrent spikingjelly LIF | Hippo-only, Spikformer SSA (T=4), relative bias | -15.27 / -13.80 / -29.75 | -19.61 | worse | 0.24 | not met |
| Z3 vs Hippo-only (criterion 2) | Neo-mixer bidir population (taus 16-128) | Hippo-only, Spikformer SSA (T=4), relative bias | -7.23 / -4.06 / -5.21 | -5.50 | worse | 0.50 | not met |
| Y2p absolute position in NSMT (population) | NSMT population + absolute position | NSMT population (T=N, psi 1) | -0.48 / -1.24 / +3.92 | +0.74 | no clear difference | 1.02 |  |
| Y3p absolute position in Neo-mixer (population) | Neo-mixer bidir population + absolute position | Neo-mixer bidir population (taus 4-32, psi 1) | -0.70 / -1.03 / -4.11 | -1.95 | worse | 1.00 |  |
| Z1p recurrence (population) | Neo-mixer bidir recurrent population f-LIF | Neo-mixer bidir population (taus 4-32, psi 1) | -9.84 / -12.36 / -9.40 | -10.54 | worse | 1.06 |  |
| W1 hippocampus T=4 + token xattn | NSMT LIF, hippocampus T=4 + token cross-attention | NSMT LIF (T=N) | +0.97 / +4.12 / -0.13 | +1.66 | better | 0.04 |  |
| W1 hippocampus T=4 + token xattn | NSMT population, hippocampus T=4 + token cross-attention | NSMT population (T=N, psi 1) | -5.93 / -3.59 / +2.69 | -2.28 | worse | 0.05 |  |
| W2 vs Hippo-only (criterion 2) | NSMT LIF, hippocampus T=4 + token cross-attention | Hippo-only, Spikformer SSA (T=4), relative bias | -4.33 / -0.02 / -2.19 | -2.18 | worse | 1.16 | not met |
| W2 vs Hippo-only (criterion 2) | NSMT population, hippocampus T=4 + token cross-attention | Hippo-only, Spikformer SSA (T=4), relative bias | -10.18 / -6.75 / -3.88 | -6.93 | worse | 1.26 | not met |
| V1 soma init | NSMT population psi 1.5 | NSMT population (T=N, psi 1) | -2.51 / +0.42 / +3.55 | +0.49 | no clear difference | 1.04 |  |
| V1 soma init | NSMT population psi 2 | NSMT population (T=N, psi 1) | -0.43 / -0.97 / +4.87 | +1.16 | no clear difference | 1.05 |  |
| V1 soma init | Neo-mixer bidir population psi 1.5 | Neo-mixer bidir population (taus 4-32, psi 1) | +2.05 / +0.49 / -1.80 | +0.24 | no clear difference | 1.01 |  |
| V1 soma init | Neo-mixer bidir population psi 2 | Neo-mixer bidir population (taus 4-32, psi 1) | +1.25 / +0.61 / -1.63 | +0.08 | no clear difference | 1.02 |  |
