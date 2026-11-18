# Mixer variants of the LIF SpkMixer, MIT-BIH

Rules and sources: module docstring of summarize_mx3.py. Seeds 42 / 7 / 2026. Tuned LIF SpkMixer settings, not re-tuned; last-epoch weights on the 30 % test file.

## Test values (seed mean)

| arm | macro-F1 % | per seed 42, 7, 2026 | accuracy % | F1 N / S / V / F / Q | rho_hippo | energy µJ | params |
|---|--:|---|--:|---|--:|--:|--:|
| base | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 |
| max | 89.24 | 88.43, 89.30, 89.98 | 97.84 | 98.86 / 77.09 / 95.16 / 76.93 / 98.16 | 0.0831 | 5.84 | 1200768 |
| mean | 91.73 | 91.42, 92.19, 91.58 | 98.53 | 99.22 / 85.45 / 96.27 / 78.76 / 98.96 | 0.0850 | 20.56 | 1200768 |
| MS | 91.96 | 91.96, 92.11, 91.81 | 98.53 | 99.23 / 85.51 / 96.28 / 79.91 / 98.86 | 0.0945 | 10.20 | 1200768 |
| subtract | 91.76 | 91.45, 91.98, 91.85 | 98.52 | 99.22 / 84.93 / 96.26 / 79.35 / 99.03 | 0.0994 | 9.35 | 1200768 |
| zero | 92.03 | 91.75, 92.10, 92.22 | 98.53 | 99.23 / 85.72 / 96.05 / 80.20 / 98.94 | 0.0969 | 8.99 | 1200768 |
| Spikformer-style | 91.78 | 92.10, 91.74, 91.50 | 98.48 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 0.1275 | 9.61 | 904488 |

## Pre-registered comparisons

| comparison | mean %p | seeds higher | per seed 42, 7, 2026 | verdict | energy A/B |
|---|--:|--:|---|---|--:|
| V1 max vs base | -2.75 | 0/3 | -3.45, -2.83, -1.96 | worse | 0.62 |
| V2 mean vs base | -0.26 | 1/3 | -0.46, +0.06, -0.36 | no clear difference | 2.18 |
| V3 MS vs base | -0.03 | 1/3 | +0.07, -0.02, -0.14 | no clear difference | 1.08 |
| V5 subtract vs base | -0.23 | 0/3 | -0.44, -0.15, -0.09 | no clear difference | 0.99 |
| V6 zero vs base | +0.04 | 1/3 | -0.14, -0.03, +0.28 | no clear difference | 0.95 |
| (report) max vs mean | -2.49 | 0/3 | -2.99, -2.89, -1.60 | worse | 0.28 |
| (report) subtract vs zero | -0.27 | 0/3 | -0.30, -0.12, -0.37 | no clear difference | 1.04 |
| (report) base vs Spikformer-style | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) max vs Spikformer-style | -2.54 | 0/3 | -3.67, -2.44, -1.52 | worse | 0.61 |
| (report) mean vs Spikformer-style | -0.05 | 2/3 | -0.68, +0.45, +0.08 | no clear difference | 2.14 |
| (report) MS vs Spikformer-style | +0.18 | 2/3 | -0.14, +0.37, +0.31 | no clear difference | 1.06 |
| (report) subtract vs Spikformer-style | -0.02 | 2/3 | -0.66, +0.24, +0.35 | no clear difference | 0.97 |
| (report) zero vs Spikformer-style | +0.24 | 2/3 | -0.35, +0.36, +0.72 | no clear difference | 0.94 |
