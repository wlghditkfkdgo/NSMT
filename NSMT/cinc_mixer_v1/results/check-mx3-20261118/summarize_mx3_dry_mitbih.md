# Mixer variants of the LIF SpkMixer, MIT-BIH

Rules and sources: module docstring of summarize_mx3.py. Seeds 42 / 7 / 2026. Tuned LIF SpkMixer settings, not re-tuned; last-epoch weights on the 30 % test file.

## Test values (seed mean)

| arm | macro-F1 % | per seed 42, 7, 2026 | accuracy % | F1 N / S / V / F / Q | rho_hippo | energy µJ | params |
|---|--:|---|--:|---|--:|--:|--:|
| base | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 |
| max | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 |
| mean | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 |
| MS | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 |
| subtract | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 |
| zero | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 |
| Spikformer-style | 91.78 | 92.10, 91.74, 91.50 | 98.48 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 0.1275 | 9.61 | 904488 |

## Pre-registered comparisons

| comparison | mean %p | seeds higher | per seed 42, 7, 2026 | verdict | energy A/B |
|---|--:|--:|---|---|--:|
| V1 max vs base | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| V2 mean vs base | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| V3 MS vs base | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| V5 subtract vs base | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| V6 zero vs base | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| (report) max vs mean | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| (report) subtract vs zero | +0.00 | 0/3 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |
| (report) base vs Spikformer-style | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) max vs Spikformer-style | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) mean vs Spikformer-style | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) MS vs Spikformer-style | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) subtract vs Spikformer-style | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) zero vs Spikformer-style | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
