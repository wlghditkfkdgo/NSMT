# Generalization batch, MIT-BIH `flif-v1-gen-mitbih` — summary

ECG framework (30 epochs, last-epoch weights on the 30 % test file); macro-F1 / accuracy mean ± sd over seeds (42, 7, 2026) (%).

| condition | n | macro-F1 | accuracy | F1 S | F1 V | F1 F | F1 Q | total µJ (FFT) | neocortex firing | params |
|---|---|---|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 3 | 92.22 ± 0.21 | 98.55 ± 0.02 | 85.28 | 96.04 | 81.38 | 99.17 | 9.55 | n/a | 904488 |
| Neo-mixer charges + |H_f - H_r|, spikeDE LIF | 3 | 91.44 ± 0.21 | 98.40 ± 0.05 | 83.88 | 96.02 | 79.35 | 98.79 | 24.65 | 0.0694 ± 0.0037 | 1347456 |
| Neo-mixer charges + |H_f - H_r|, f-LIF | 3 | 91.44 ± 0.31 | 98.38 ± 0.02 | 83.84 | 96.03 | 79.44 | 98.76 | 24.78 | 0.0550 ± 0.0016 | 1347456 |
| Neo-mixer charges + |H_f - H_r|, population LIF (psi 0.5) | 3 | 91.55 ± 0.21 | 98.37 ± 0.04 | 84.50 | 95.86 | 79.84 | 98.40 | 25.40 | 0.0378 ± 0.0010 | 1353600 |
| Neo-mixer charges + |H_f - H_r|, population f-LIF (psi 1) | 3 | 91.25 ± 0.24 | 98.37 ± 0.03 | 84.23 | 95.87 | 78.41 | 98.57 | 26.82 | 0.0375 ± 0.0008 | 1353600 |
| Neo-mixer OR, spikeDE LIF | 3 | 91.14 ± 0.08 | 98.32 ± 0.04 | 83.64 | 95.63 | 78.75 | 98.58 | 2.15 | 0.0746 ± 0.0013 | 904320 |
| Neo-mixer OR, f-LIF | 3 | 91.13 ± 0.21 | 98.30 ± 0.04 | 82.81 | 95.66 | 79.36 | 98.73 | 2.25 | 0.0571 ± 0.0052 | 904320 |
| Neo-mixer OR, population LIF (psi 0.5) | 3 | 91.27 ± 0.26 | 98.30 ± 0.02 | 83.56 | 95.76 | 79.50 | 98.45 | 3.00 | 0.0901 ± 0.0010 | 910464 |
| Neo-mixer OR, population f-LIF (psi 1) | 3 | 91.19 ± 0.45 | 98.30 ± 0.04 | 83.03 | 95.79 | 79.46 | 98.56 | 4.39 | 0.0794 ± 0.0004 | 910464 |

## Pre-registered comparisons (macro-F1, %p, paired by seed)

| question | A | B | A − B per seed | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|
| G1 linmx LIF vs Spikformer (primary, criterion 2) | Neo-mixer charges + |H_f - H_r|, spikeDE LIF | Spikformer-style SSA (T=4) | -0.84 / -1.05 / -0.47 | -0.78 | worse | 2.58 | not met |
| G2 linmx vs Spikformer (criterion 2) | Neo-mixer charges + |H_f - H_r|, f-LIF | Spikformer-style SSA (T=4) | -0.67 / -0.82 / -0.87 | -0.78 | worse | 2.59 | not met |
| G2 linmx vs Spikformer (criterion 2) | Neo-mixer charges + |H_f - H_r|, population LIF (psi 0.5) | Spikformer-style SSA (T=4) | -1.10 / -0.53 / -0.39 | -0.67 | worse | 2.66 | not met |
| G2 linmx vs Spikformer (criterion 2) | Neo-mixer charges + |H_f - H_r|, population f-LIF (psi 1) | Spikformer-style SSA (T=4) | -1.10 / -0.80 / -1.02 | -0.98 | worse | 2.81 | not met |
| G3 linmx vs OR (fusion) | Neo-mixer charges + |H_f - H_r|, spikeDE LIF | Neo-mixer OR, spikeDE LIF | +0.51 / -0.04 / +0.41 | +0.29 | no clear difference | 11.49 |  |
| G3 linmx vs OR (fusion) | Neo-mixer charges + |H_f - H_r|, f-LIF | Neo-mixer OR, f-LIF | +0.43 / +0.53 / -0.03 | +0.31 | no clear difference | 11.03 |  |
| G3 linmx vs OR (fusion) | Neo-mixer charges + |H_f - H_r|, population LIF (psi 0.5) | Neo-mixer OR, population LIF (psi 0.5) | -0.10 / +0.29 / +0.64 | +0.28 | no clear difference | 8.47 |  |
| G3 linmx vs OR (fusion) | Neo-mixer charges + |H_f - H_r|, population f-LIF (psi 1) | Neo-mixer OR, population f-LIF (psi 1) | -0.26 / +0.74 / -0.31 | +0.06 | no clear difference | 6.10 |  |
| G4 OR population LIF vs Spikformer (criterion 2) | Neo-mixer OR, population LIF (psi 0.5) | Spikformer-style SSA (T=4) | -1.00 / -0.82 / -1.03 | -0.95 | worse | 0.31 | not met |
| G5 (report) OR vs Spikformer (criterion 2) | Neo-mixer OR, spikeDE LIF | Spikformer-style SSA (T=4) | -1.35 / -1.01 / -0.87 | -1.08 | worse | 0.22 | not met |
| G5 (report) OR vs Spikformer (criterion 2) | Neo-mixer OR, f-LIF | Spikformer-style SSA (T=4) | -1.10 / -1.34 / -0.84 | -1.09 | worse | 0.24 | not met |
| G5 (report) OR vs Spikformer (criterion 2) | Neo-mixer OR, population f-LIF (psi 1) | Spikformer-style SSA (T=4) | -0.84 / -1.55 / -0.71 | -1.03 | worse | 0.46 | not met |
