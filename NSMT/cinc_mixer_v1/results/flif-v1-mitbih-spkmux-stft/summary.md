# STFT band test on the SpkMixer, MIT-BIH

Rules and sources: module docstring of summarize_stft.py. Seeds 42 / 7 / 2026. Tuned LIF SpkMixer settings, not re-tuned; last-epoch weights on the 30 % test file.

STFT cost not counted by the hooks (upper estimate, 11 frames x 32 x 32 at E_MAC): 0.052 µJ per beat.

## Test values (seed mean)

| form | macro-F1 % | per seed 42, 7, 2026 | accuracy % | F1 N / S / V / F / Q | rho_hippo | energy µJ | params |
|---|--:|---|--:|---|--:|--:|--:|
| LIF rep | 91.99 | 91.89, 92.13, 91.94 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 0.1001 | 9.44 | 1200768 |
| LIF token | 91.10 | 91.22, 91.47, 90.60 | 98.32 | 99.12 / 83.85 / 95.54 / 78.41 / 98.57 | 0.0947 | 2.59 | 1200768 |
| A rep | 90.22 | 90.14, 90.32, 90.21 | 98.13 | 99.02 / 82.38 / 94.79 / 76.57 / 98.35 | 0.1073 | 8.25 | 1191552 |
| A token | 88.29 | 88.19, 88.47, 88.20 | 97.66 | 98.75 / 78.09 / 93.60 / 73.29 / 97.71 | 0.1087 | 2.29 | 1191552 |
| POP rep | 91.40 | 91.50, 91.73, 90.96 | 98.49 | 99.20 / 85.24 / 96.28 / 77.39 / 98.88 | 0.1027 | 19.98 | 1214592 |
| POP token | 89.95 | 89.99, 90.25, 89.62 | 97.93 | 98.90 / 81.37 / 94.23 / 77.18 / 98.09 | 0.1004 | 5.29 | 1214592 |
| B rep | 89.96 | 89.63, 90.37, 89.87 | 98.09 | 98.99 / 82.03 / 94.68 / 75.84 / 98.24 | 0.1288 | 21.93 | 1216896 |
| B token | 86.75 | 86.46, 87.05, 86.73 | 97.08 | 98.43 / 75.60 / 90.96 / 72.37 / 96.38 | 0.1161 | 5.74 | 1216896 |
| Spikformer-style | 91.78 | 92.10, 91.74, 91.50 | 98.48 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 0.1275 | 9.61 | 904488 |

## Pre-registered comparisons

| comparison | mean %p | seeds higher | per seed 42, 7, 2026 | verdict | energy A/B |
|---|--:|--:|---|---|--:|
| S1 A rep vs LIF rep | -1.76 | 0/3 | -1.74, -1.81, -1.73 | worse | 0.87 |
| S2 B rep vs POP rep | -1.44 | 0/3 | -1.87, -1.37, -1.09 | worse | 1.10 |
| S3 A rep vs B rep | +0.27 | 2/3 | +0.52, -0.05, +0.34 | no clear difference | 0.38 |
| S1 A token vs LIF token | -2.81 | 0/3 | -3.03, -3.00, -2.40 | worse | 0.89 |
| S2 B token vs POP token | -3.20 | 0/3 | -3.52, -3.20, -2.89 | worse | 1.09 |
| S3 A token vs B token | +1.54 | 3/3 | +1.72, +1.42, +1.47 | better | 0.40 |
| S4 LIF token vs LIF rep | -0.89 | 0/3 | -0.66, -0.66, -1.34 | worse | 0.27 |
| S4 A token vs A rep | -1.94 | 0/3 | -1.96, -1.85, -2.01 | worse | 0.28 |
| S4 POP token vs POP rep | -1.44 | 0/3 | -1.52, -1.48, -1.33 | worse | 0.26 |
| S4 B token vs B rep | -3.21 | 0/3 | -3.17, -3.32, -3.13 | worse | 0.26 |
| (report) POP rep vs LIF rep | -0.59 | 0/3 | -0.38, -0.39, -0.98 | worse | 2.12 |
| (report) POP token vs LIF token | -1.14 | 0/3 | -1.24, -1.22, -0.97 | worse | 2.04 |
| (report) LIF rep vs Spikformer-style | +0.20 | 2/3 | -0.22, +0.39, +0.44 | no clear difference | 0.98 |
| (report) A rep vs Spikformer-style | -1.56 | 0/3 | -1.96, -1.42, -1.29 | worse | 0.86 |
| (report) POP rep vs Spikformer-style | -0.38 | 0/3 | -0.60, -0.01, -0.54 | no clear difference | 2.08 |
| (report) B rep vs Spikformer-style | -1.83 | 0/3 | -2.47, -1.37, -1.63 | worse | 2.28 |
| (report) LIF token vs Spikformer-style | -0.68 | 0/3 | -0.88, -0.27, -0.90 | worse | 0.27 |
| (report) A token vs Spikformer-style | -3.49 | 0/3 | -3.92, -3.27, -3.30 | worse | 0.24 |
| (report) POP token vs Spikformer-style | -1.83 | 0/3 | -2.12, -1.49, -1.88 | worse | 0.55 |
| (report) B token vs Spikformer-style | -5.03 | 0/3 | -5.64, -4.69, -4.77 | worse | 0.60 |

Layout choice (item 3): not made on MIT-BIH (no validation split); both layouts are shown above.
