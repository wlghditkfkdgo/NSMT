# STFT band test on the SpkMixer, ETT

Rules and sources: module docstring of summarize_stft.py. Seeds 42 / 7 / 2026. Tuned LIF SpkMixer settings, not re-tuned.

STFT cost not counted by the hooks (upper estimate, 24 frames x 8 x 8 x 7 channels at E_MAC): 0.049 µJ per window.

## Test MSE (seed mean) and energy

| form | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean | energy µJ (4-cell mean) | params (ETTh2 H96) |
|---|--:|--:|--:|--:|--:|--:|--:|
| LIF rep | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 10.28 | 169280 |
| LIF token | 0.2997 | 0.4302 | 0.1784 | 0.4040 | 0.3281 | 2.50 | 169280 |
| A rep | 0.2911 | 0.4234 | 0.1801 | 0.4043 | 0.3247 | 8.88 | 168896 |
| A token | 0.2987 | 0.4286 | 0.1814 | 0.4042 | 0.3282 | 2.16 | 168896 |
| POP rep | 0.2910 | 0.4134 | 0.1769 | 0.4033 | 0.3212 | 28.50 | 170944 |
| POP token | 0.3049 | 0.4363 | 0.1783 | 0.4019 | 0.3304 | 7.08 | 170944 |
| B rep | 0.2905 | 0.4188 | 0.1802 | 0.4059 | 0.3238 | 29.73 | 171328 |
| B token | 0.3145 | 0.4608 | 0.1814 | 0.4068 | 0.3409 | 7.45 | 171328 |
| Spikformer-style | 0.2903 | 0.4223 | 0.1767 | 0.4044 | 0.3234 | 10.48 | 169656 |

## Pre-registered comparisons

| comparison | R % | lower pairs | verdict | energy A/B | per cell r % | MAE R % |
|---|--:|--:|---|--:|---|--:|
| S1 A rep vs LIF rep | +0.16 | 4/12 | no clear difference | 0.86 | -1.36, +0.38, +1.42, +0.21 | +0.28 |
| S2 B rep vs POP rep | +0.89 | 2/12 | no clear difference | 1.04 | -0.18, +1.30, +1.82, +0.64 | +0.83 |
| S3 A rep vs B rep | +0.22 | 8/12 | no clear difference | 0.30 | +0.24, +1.10, -0.05, -0.40 | +0.18 |
| S1 A token vs LIF token | +0.27 | 6/12 | no clear difference | 0.87 | -0.33, -0.36, +1.70, +0.05 | +0.24 |
| S2 B token vs POP token | +2.92 | 0/12 | worse | 1.05 | +3.17, +5.61, +1.69, +1.22 | +2.33 |
| S3 A token vs B token | -3.15 | 9/12 | better | 0.29 | -5.02, -6.99, +0.05, -0.62 | -2.22 |
| S4 LIF token vs LIF rep | +1.04 | 0/12 | worse | 0.24 | +1.54, +1.99, +0.49, +0.14 | +0.85 |
| S4 A token vs A rep | +1.16 | 2/12 | worse | 0.24 | +2.63, +1.24, +0.77, -0.02 | +0.82 |
| S4 POP token vs POP rep | +2.69 | 3/12 | worse | 0.25 | +4.77, +5.55, +0.79, -0.36 | +1.80 |
| S4 B token vs B rep | +4.80 | 1/12 | worse | 0.25 | +8.29, +10.03, +0.67, +0.21 | +3.34 |
| (report) POP rep vs LIF rep | -0.94 | 10/12 | no clear difference | 2.78 | -1.41, -1.98, -0.34, -0.02 | -0.71 |
| (report) POP token vs LIF token | +0.65 | 5/12 | no clear difference | 2.84 | +1.72, +1.44, -0.04, -0.53 | +0.21 |
| (report) LIF rep vs Spikformer-style | +0.45 | 3/12 | no clear difference | 0.98 | +1.67, -0.12, +0.47, -0.23 | +0.45 |
| (report) A rep vs Spikformer-style | +0.60 | 3/12 | no clear difference | 0.85 | +0.27, +0.26, +1.89, -0.02 | +0.73 |
| (report) POP rep vs Spikformer-style | -0.50 | 8/12 | no clear difference | 2.73 | +0.23, -2.11, +0.13, -0.26 | -0.27 |
| (report) B rep vs Spikformer-style | +0.38 | 5/12 | no clear difference | 2.85 | +0.04, -0.83, +1.95, +0.38 | +0.55 |
| (report) LIF token vs Spikformer-style | +1.50 | 1/12 | worse | 0.24 | +3.24, +1.87, +0.96, -0.09 | +1.30 |
| (report) A token vs Spikformer-style | +1.76 | 1/12 | worse | 0.21 | +2.90, +1.48, +2.68, -0.04 | +1.55 |
| (report) POP token vs Spikformer-style | +2.16 | 3/12 | worse | 0.68 | +5.01, +3.32, +0.92, -0.62 | +1.52 |
| (report) B token vs Spikformer-style | +5.17 | 1/12 | worse | 0.71 | +8.34, +9.12, +2.63, +0.60 | +3.91 |

Pairs r % (test MSE), seed 42, 7, 2026 per cell:

- S1 A rep vs LIF rep: ETTh2 H96 -2.03, +0.60, -2.66; ETTh2 H720 +1.31, +1.71, -1.89; ETTm2 H96 +2.13, +1.26, +0.86; ETTm2 H720 +0.31, +0.38, -0.05
- S2 B rep vs POP rep: ETTh2 H96 +1.31, -0.93, -0.93; ETTh2 H720 +1.62, +1.48, +0.81; ETTm2 H96 +2.11, +1.43, +1.92; ETTm2 H720 +0.36, +0.78, +0.77
- S3 A rep vs B rep: ETTh2 H96 -0.07, +1.42, -0.64; ETTh2 H720 +3.22, +1.49, -1.41; ETTm2 H96 +0.33, -0.15, -0.34; ETTm2 H720 -0.40, -0.26, -0.52
- S1 A token vs LIF token: ETTh2 H96 -1.51, -0.85, +1.37; ETTh2 H720 -2.39, +1.31, -0.01; ETTm2 H96 +2.21, +1.94, +0.96; ETTm2 H720 +0.33, -0.14, -0.03
- S2 B token vs POP token: ETTh2 H96 +3.83, +2.79, +2.88; ETTh2 H720 +4.19, +7.60, +5.05; ETTm2 H96 +1.66, +1.43, +1.99; ETTm2 H720 +0.26, +1.75, +1.65
- S3 A token vs B token: ETTh2 H96 -5.97, -4.90, -4.20; ETTh2 H720 -6.10, -8.16, -6.71; ETTm2 H96 +0.45, +0.20, -0.50; ETTm2 H720 +0.26, -1.16, -0.97
- S4 LIF token vs LIF rep: ETTh2 H96 +0.92, +2.35, +1.37; ETTh2 H720 +3.47, +1.36, +1.14; ETTm2 H96 +0.50, +0.47, +0.49; ETTm2 H720 +0.03, +0.22, +0.18
- S4 A token vs A rep: ETTh2 H96 +1.45, +0.88, +5.56; ETTh2 H720 -0.31, +0.96, +3.08; ETTm2 H96 +0.58, +1.14, +0.59; ETTm2 H720 +0.04, -0.29, +0.20
- S4 POP token vs POP rep: ETTh2 H96 +5.20, +3.68, +5.42; ETTh2 H720 +6.87, +5.22, +4.54; ETTm2 H96 +0.91, +0.78, +0.68; ETTm2 H720 -0.53, -0.34, -0.22
- S4 B token vs B rep: ETTh2 H96 +7.82, +7.58, +9.48; ETTh2 H720 +9.58, +11.57, +8.94; ETTm2 H96 +0.46, +0.79, +0.75; ETTm2 H720 -0.63, +0.62, +0.65

## Layout choice by validation (item 3; test not used)

| form | mean best validation L1, rep | token | token / rep - 1 % | chosen | test R % token vs rep |
|---|--:|--:|--:|---|--:|
| LIF | 0.36145 | 0.36446 | +0.83 | rep | +1.04 |
| A | 0.36302 | 0.36624 | +0.89 | rep | +1.16 |
| POP | 0.36151 | 0.36923 | +2.14 | rep | +2.69 |
| B | 0.36530 | 0.37339 | +2.21 | rep | +4.80 |

## Report only: stopping and firing rates (seed mean)

| cell | form | best epoch (seeds) | epochs run (seeds) | last-epoch MSE | rho_hippo | energy µJ |
|---|---|---|---|--:|--:|--:|
| ETTh2 H96 | LIF rep | 3, 1, 6 | 14, 12, 17 | 0.2960 | 0.0704 | 9.06 |
| ETTh2 H96 | LIF token | 5, 2, 3 | 16, 13, 14 | 0.3036 | 0.0600 | 2.20 |
| ETTh2 H96 | A rep | 1, 1, 2 | 12, 12, 13 | 0.2911 | 0.0757 | 7.80 |
| ETTh2 H96 | A token | 4, 5, 6 | 15, 16, 17 | 0.3053 | 0.0739 | 1.93 |
| ETTh2 H96 | POP rep | 0, 1, 1 | 11, 12, 12 | 0.2892 | 0.0715 | 26.63 |
| ETTh2 H96 | POP token | 9, 5, 6 | 20, 16, 17 | 0.3044 | 0.0742 | 6.64 |
| ETTh2 H96 | B rep | 1, 1, 2 | 12, 12, 13 | 0.3038 | 0.0794 | 27.98 |
| ETTh2 H96 | B token | 12, 21, 6 | 23, 32, 17 | 0.3147 | 0.0850 | 7.02 |
| ETTh2 H720 | LIF rep | 1, 0, 2 | 12, 11, 13 | 0.4191 | 0.0629 | 10.67 |
| ETTh2 H720 | LIF token | 3, 0, 2 | 14, 11, 13 | 0.4245 | 0.0554 | 2.48 |
| ETTh2 H720 | A rep | 5, 5, 2 | 16, 16, 13 | 0.4376 | 0.0809 | 9.43 |
| ETTh2 H720 | A token | 3, 1, 4 | 14, 12, 15 | 0.4305 | 0.0667 | 2.21 |
| ETTh2 H720 | POP rep | 1, 0, 2 | 12, 11, 13 | 0.4201 | 0.0678 | 28.32 |
| ETTh2 H720 | POP token | 3, 3, 4 | 14, 14, 15 | 0.4401 | 0.0727 | 7.01 |
| ETTh2 H720 | B rep | 1, 0, 0 | 12, 11, 11 | 0.4479 | 0.0758 | 29.78 |
| ETTh2 H720 | B token | 4, 9, 4 | 15, 20, 15 | 0.4669 | 0.0817 | 7.37 |
| ETTm2 H96 | LIF rep | 25, 11, 7 | 36, 22, 18 | 0.1773 | 0.0832 | 9.89 |
| ETTm2 H96 | LIF token | 21, 17, 15 | 32, 28, 26 | 0.1786 | 0.0822 | 2.47 |
| ETTm2 H96 | A rep | 8, 11, 9 | 19, 22, 20 | 0.1802 | 0.0806 | 8.55 |
| ETTm2 H96 | A token | 11, 13, 19 | 22, 24, 30 | 0.1813 | 0.0792 | 2.13 |
| ETTm2 H96 | POP rep | 8, 5, 6 | 19, 16, 17 | 0.1768 | 0.0799 | 28.62 |
| ETTm2 H96 | POP token | 15, 34, 15 | 26, 45, 26 | 0.1785 | 0.0777 | 7.11 |
| ETTm2 H96 | B rep | 5, 5, 6 | 16, 16, 17 | 0.1796 | 0.0822 | 29.98 |
| ETTm2 H96 | B token | 13, 7, 26 | 24, 18, 37 | 0.1808 | 0.0814 | 7.51 |
| ETTm2 H720 | LIF rep | 2, 4, 2 | 13, 15, 13 | 0.4096 | 0.0744 | 11.49 |
| ETTm2 H720 | LIF token | 3, 6, 4 | 14, 17, 15 | 0.4065 | 0.0717 | 2.84 |
| ETTm2 H720 | A rep | 2, 3, 2 | 13, 14, 13 | 0.4128 | 0.0738 | 9.74 |
| ETTm2 H720 | A token | 2, 3, 2 | 13, 14, 13 | 0.4063 | 0.0677 | 2.37 |
| ETTm2 H720 | POP rep | 2, 1, 1 | 13, 12, 12 | 0.4104 | 0.0705 | 30.43 |
| ETTm2 H720 | POP token | 3, 9, 9 | 14, 20, 20 | 0.4017 | 0.0756 | 7.56 |
| ETTm2 H720 | B rep | 1, 1, 1 | 12, 12, 12 | 0.4161 | 0.0725 | 31.20 |
| ETTm2 H720 | B token | 3, 1, 1 | 14, 12, 12 | 0.4049 | 0.0741 | 7.91 |
