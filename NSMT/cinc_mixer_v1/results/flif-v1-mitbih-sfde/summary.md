# Part 1, MIT-BIH: LIF SpkMixer vs the Spikformer-style model with the same spikeDE LIF (SF-de)

Rules and sources: module docstring of summarize_pn.py. Seeds 42 / 7 / 2026; last-epoch weights on the 30 % test file.

## Test values (seed mean)

| arm | macro-F1 % | per seed 42, 7, 2026 | accuracy % | F1 N / S / V / F / Q | rho_hippo | energy µJ | params | time s (seeds) |
|---|--:|---|--:|---|--:|--:|--:|---|
| LIF SpkMixer | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 | 1311, 1297, 1311 |
| SF-de | 91.94 | 91.76, 92.50, 91.57 | 98.53 | 99.23 / 85.14 / 96.09 / 80.23 / 99.03 | 0.1165 | 9.45 | 904488 | 926, 1119, 926 |
| SF-sj | 91.78 | 92.10, 91.74, 91.50 | 98.48 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 0.1275 | 9.61 | 904488 | 420, 408, 410 |

## Comparisons

| comparison | mean %p | seeds higher | per seed 42, 7, 2026 | verdict | energy A/B |
|---|--:|--:|---|---|--:|
| A1 LIF SpkMixer vs SF-de | +0.04 | 2/3 | +0.13, -0.37, +0.37 | no clear difference | 1.00 |
| (report) SF-de vs SF-sj | +0.16 | 2/3 | -0.35, +0.76, +0.07 | no clear difference | 0.98 |
| (report) LIF SpkMixer vs SF-sj | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
