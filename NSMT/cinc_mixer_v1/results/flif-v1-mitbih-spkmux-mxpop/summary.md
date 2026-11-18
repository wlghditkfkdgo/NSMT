# Mixer-only population, MIT-BIH (user request 11-19): population neuron only in the directional stacks + fusion

Rules and sources: module docstring of summarize_pn.py. Seeds 42 / 7 / 2026; last-epoch weights on the 30 % test file.

## Test values (seed mean)

| arm | macro-F1 % | per seed 42, 7, 2026 | accuracy % | F1 N / S / V / F / Q | rho_hippo | energy µJ | params | time s (seeds) |
|---|--:|---|--:|---|--:|--:|--:|---|
| LIF | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 | 1311, 1297, 1311 |
| LIF+mean | 91.73 | 91.42, 92.19, 91.58 | 98.53 | 99.22 / 85.45 / 96.27 / 78.76 / 98.96 | 0.0850 | 20.56 | 1200768 | 1190, 834, 1012 |
| pop LIF all | 91.40 | 91.50, 91.73, 90.96 | 98.49 | 99.20 / 85.24 / 96.28 / 77.39 / 98.88 | 0.1027 | 19.98 | 1214592 | 5979, 5971, 5971 |
| pop LIF mixer | 92.17 | 92.16, 92.16, 92.19 | 98.56 | 99.25 / 85.81 / 96.21 / 80.71 / 98.87 | 0.1132 | 17.43 | 1209984 | 3558, 4484, 4444 |
| pop LIF mixer+mean | 91.81 | 92.02, 91.72, 91.68 | 98.51 | 99.21 / 85.42 / 96.31 / 79.33 / 98.77 | 0.0834 | 22.64 | 1209984 | 1796, 1256, 1466 |
| pop f-LIF mixer | 91.97 | 91.66, 92.43, 91.83 | 98.58 | 99.27 / 85.65 / 96.29 / 79.67 / 98.99 | 0.1144 | 25.34 | 1209984 | 4702, 4664, 3919 |
| pop f-LIF mixer+mean | 91.99 | 92.25, 91.83, 91.89 | 98.54 | 99.24 / 85.51 / 96.11 / 80.15 / 98.94 | 0.0892 | 24.77 | 1209984 | 2552, 2554, 2224 |
| SF-de | 91.94 | 91.76, 92.50, 91.57 | 98.53 | 99.23 / 85.14 / 96.09 / 80.23 / 99.03 | 0.1165 | 9.45 | 904488 | 926, 1119, 926 |
| SF-sj | 91.78 | 92.10, 91.74, 91.50 | 98.48 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 0.1275 | 9.61 | 904488 | 420, 408, 410 |

## Comparisons

| comparison | mean %p | seeds higher | per seed 42, 7, 2026 | verdict | energy A/B |
|---|--:|--:|---|---|--:|
| B1 pop LIF mixer vs LIF | +0.18 | 3/3 | +0.27, +0.04, +0.25 | no clear difference | 1.85 |
| B3 pop LIF mixer vs pop LIF all | +0.77 | 3/3 | +0.65, +0.43, +1.23 | better | 0.87 |
| C3 pop LIF mixer+mean vs pop LIF mixer | -0.36 | 0/3 | -0.13, -0.44, -0.51 | no clear difference | 1.30 |
| C5 pop f-LIF mixer+mean vs pop f-LIF mixer | +0.02 | 2/3 | +0.59, -0.59, +0.05 | no clear difference | 0.98 |
| (report) pop f-LIF mixer vs pop LIF mixer | -0.20 | 1/3 | -0.49, +0.26, -0.36 | no clear difference | 1.45 |
| (report, mean) pop LIF mixer+mean vs LIF+mean | +0.08 | 2/3 | +0.60, -0.46, +0.10 | no clear difference | 1.10 |
| (report) LIF vs SF-de | +0.04 | 2/3 | +0.13, -0.37, +0.37 | no clear difference | 1.00 |
| (report) LIF+mean vs SF-de | -0.21 | 1/3 | -0.33, -0.31, +0.01 | no clear difference | 2.18 |
| (report) pop LIF mixer vs SF-de | +0.23 | 2/3 | +0.40, -0.34, +0.62 | no clear difference | 1.84 |
| (report) pop LIF mixer+mean vs SF-de | -0.13 | 2/3 | +0.27, -0.78, +0.11 | no clear difference | 2.40 |
| (report) pop f-LIF mixer vs SF-de | +0.03 | 1/3 | -0.09, -0.08, +0.26 | no clear difference | 2.68 |
| (report) pop f-LIF mixer+mean vs SF-de | +0.05 | 2/3 | +0.49, -0.67, +0.31 | no clear difference | 2.62 |
| (report) SF-de vs SF-sj | +0.16 | 2/3 | -0.35, +0.76, +0.07 | no clear difference | 0.98 |

## Paper bar vs SF-de (user decision 11-18: mean macro-F1 higher and energy A/B < 1, sign only)

Existing energy = total_fft_uJ_corr (integer rule). Pooled-integer rule (mean arms only; proposal awaiting the user's approval): a Linear input whose T_h-fold is integer counts sum(T_h x) ACs (energy_pool_mitbih.py / check-pool-energy-20261119); SF-de unchanged.

| arm | mean %p vs SF-de | higher (> 0) | energy A/B | paper bar | energy µJ pooled rule | energy A/B pooled rule | paper bar pooled rule |
|---|--:|---|--:|---|--:|--:|---|
| LIF | +0.042 | yes | 0.9993 | met | - | - | - |
| LIF+mean | -0.213 | no | 2.1772 | not met | 6.00 | 0.6356 | not met |
| pop LIF mixer | +0.226 | yes | 1.8449 | not met | - | - | - |
| pop LIF mixer+mean | -0.133 | no | 2.3971 | not met | 8.06 | 0.8537 | not met |
| pop f-LIF mixer | +0.030 | yes | 2.6832 | not met | - | - | - |
| pop f-LIF mixer+mean | +0.046 | yes | 2.6224 | not met | 10.29 | 1.0893 | not met |
