# Membrane shortcut test, MIT-BIH (LIF only)

Rules and sources: module docstring of summarize_ms.py. Seeds 42 / 7 / 2026. Tuned settings, not re-tuned for the shortcut; last-epoch weights on the 30 % test file.

## Test values (seed mean)

| model | macro-F1 % | accuracy % | F1 N / S / V / F / Q | rho_hippo | energy µJ |
|---|--:|--:|---|--:|--:|
| SpkMixer (ADD) | 91.99 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 |
| SpkMixer + MS | 91.89 | 98.55 | 99.24 / 85.57 / 96.44 / 79.24 / 98.98 | 0.0840 | 9.94 |
| Spikformer-style | 91.78 | 98.48 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 0.1275 | 9.61 |
| Spikformer-style + MS | 91.21 | 98.32 | 99.12 / 83.60 / 95.67 / 78.98 / 98.70 | 0.1050 | 8.06 |

## Pre-registered comparisons

| comparison | mean %p | seeds higher | per seed 42, 7, 2026 | verdict | energy A/B |
|---|--:|--:|---|---|--:|
| M1 SpkMixer + MS vs SpkMixer | -0.09 | 1/3 | +0.33, -0.13, -0.47 | no clear difference | 1.05 |
| M2 Spikformer-style + MS vs Spikformer-style | -0.57 | 0/3 | -0.68, -0.36, -0.67 | worse | 0.84 |
| M3 SpkMixer + MS vs Spikformer-style + MS | +0.68 | 3/3 | +0.78, +0.61, +0.65 | better | 1.23 |
| M4 (report only) SpkMixer + MS vs Spikformer-style | +0.11 | 2/3 | +0.11, +0.26, -0.03 | no clear difference | 1.03 |
| (reference) SpkMixer vs Spikformer-style | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
