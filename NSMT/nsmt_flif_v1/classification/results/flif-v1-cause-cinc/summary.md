# Cause analysis without training `flif-v1-cause-cinc` (CINC2017 fold 0, 90-epoch weights) — summary

Best-validation weights, test windows. Values: mean (seed 42 / 7 / 2026). Pre-registered rules: docs/MODI_PROJECT_LOG.md (2026-10-28) and the docstring of summarize_cause.py.

Runs found: 30 of 30; missing: none; failed checks: none.

## B models: measurements

| measurement | B, memoryless LIF | B, T = N LIF | B, T = N f-LIF | B, T = N population LIF | B, T = N population f-LIF |
|---|---|---|---|---|---|
| test score (%) | 66.04 (65.74 / 66.26 / 66.12) | 65.49 (65.89 / 65.58 / 65.01) | 64.25 (65.66 / 62.70 / 64.38) | 65.38 (66.02 / 66.16 / 63.95) | 65.28 (63.82 / 64.95 / 67.08) |
| drop: neocortex of the next window (%p) | 4.30 (3.34 / 4.98 / 4.57) | 11.16 (11.25 / 10.72 / 11.51) | 10.43 (10.83 / 10.90 / 9.58) | 7.15 (7.29 / 7.05 / 7.11) | 5.16 (5.50 / 3.65 / 6.34) |
| drop: neocortex zeroed (%p) | 2.55 (1.20 / 4.17 / 2.26) | 17.42 (15.27 / 18.58 / 18.40) | 15.11 (11.19 / 15.78 / 18.36) | 7.26 (5.24 / 10.67 / 5.87) | 9.50 (5.03 / 13.50 / 9.97) |
| drop: history scramble (%p) | 0.00 (0.00 / 0.00 / 0.00) | 6.13 (6.74 / 6.12 / 5.55) | 6.73 (7.38 / 4.67 / 8.13) | 4.34 (4.87 / 4.18 / 3.98) | 2.28 (3.06 / -0.10 / 3.87) |
| neocortex term share of the attention output | 0.288 (0.314 / 0.249 / 0.301) | 0.419 (0.440 / 0.402 / 0.416) | 0.424 (0.435 / 0.415 / 0.423) | 0.454 (0.417 / 0.509 / 0.435) | 0.348 (0.368 / 0.338 / 0.336) |
| H1 readout R² own patch (lag 0) | 0.318 (0.317 / 0.328 / 0.310) | 0.242 (0.237 / 0.249 / 0.240) | 0.343 (0.336 / 0.360 / 0.333) | -0.005 (-0.005 / -0.006 / -0.006) | 0.007 (0.005 / 0.007 / 0.008) |
| H1 readout R² patch lag 2 | 0.054 (0.056 / 0.053 / 0.052) | 0.101 (0.097 / 0.100 / 0.106) | 0.159 (0.162 / 0.149 / 0.165) | -0.003 (-0.003 / -0.003 / -0.003) | 0.002 (0.001 / 0.001 / 0.003) |
| H1 readout R² R-peak in token | 0.115 (0.115 / 0.119 / 0.110) | 0.142 (0.131 / 0.160 / 0.135) | 0.235 (0.228 / 0.255 / 0.223) | 0.022 (0.027 / 0.018 / 0.020) | 0.038 (0.030 / 0.047 / 0.038) |
| H1 readout R² last RR interval | -0.013 (-0.010 / -0.014 / -0.014) | 0.033 (0.034 / 0.033 / 0.031) | 0.045 (0.043 / 0.054 / 0.039) | -0.041 (-0.029 / -0.057 / -0.039) | -0.026 (-0.022 / -0.023 / -0.033) |
| H1 adjacent-token cosine, output | 0.213 (0.222 / 0.201 / 0.215) | 0.408 (0.431 / 0.398 / 0.393) | 0.269 (0.256 / 0.298 / 0.253) | 0.450 (0.446 / 0.466 / 0.436) | 0.309 (0.299 / 0.311 / 0.317) |
| H1 adjacent-token cosine, input | 0.020 (0.020 / 0.020 / 0.020) | 0.020 (0.020 / 0.020 / 0.020) | 0.020 (0.020 / 0.020 / 0.020) | 0.020 (0.020 / 0.020 / 0.020) | 0.020 (0.020 / 0.020 / 0.020) |
| H1 effective number of neocortex keys | 71.2 (66.0 / 71.7 / 76.0) | 70.9 (69.9 / 70.7 / 72.2) | 73.6 (73.5 / 72.4 / 75.0) | 70.0 (70.7 / 70.5 / 68.7) | 69.8 (67.8 / 69.8 / 71.8) |
| H1 effective number of own keys | 66.8 (62.0 / 66.6 / 71.7) | 66.9 (66.5 / 66.1 / 68.1) | 69.5 (69.8 / 68.2 / 70.5) | 62.3 (62.3 / 63.0 / 61.5) | 62.4 (60.2 / 62.4 / 64.6) |
| H2 output firing rate | 0.0278 (0.0288 / 0.0275 / 0.0270) | 0.0450 (0.0420 / 0.0458 / 0.0472) | 0.0361 (0.0370 / 0.0363 / 0.0350) | 0.0943 (0.0907 / 0.1008 / 0.0914) | 0.0706 (0.0720 / 0.0684 / 0.0714) |
| H2 rate first 5 / steady | 3.051 (3.180 / 2.966 / 3.008) | 2.041 (2.054 / 2.010 / 2.058) | 1.815 (1.872 / 1.602 / 1.971) | 0.041 (0.042 / 0.049 / 0.033) | 0.553 (0.584 / 0.521 / 0.554) |
| H2 rate slope per 100 tokens (rel.) | -0.161 (-0.180 / -0.131 / -0.172) | -0.132 (-0.126 / -0.141 / -0.129) | -0.095 (-0.110 / -0.087 / -0.088) | 0.240 (0.216 / 0.229 / 0.274) | 0.434 (0.385 / 0.459 / 0.457) |
| H2 readout first 5 / steady | 1.902 (1.905 / 1.862 / 1.939) | 1.178 (1.202 / 1.176 / 1.156) | 1.209 (1.206 / 1.190 / 1.231) | 0.498 (0.535 / 0.377 / 0.582) | -11.715 (-6.191 / -11.220 / -17.735) |
| H3 distance from init, neocortex | 3.058 (3.104 / 3.092 / 2.979) | 3.161 (2.953 / 3.339 / 3.189) | 3.166 (3.323 / 3.002 / 3.172) | 1.121 (1.114 / 1.141 / 1.107) | 0.573 (0.484 / 0.616 / 0.618) |
| H3 distance from init, neocortex keys/values | 2.941 (3.016 / 2.914 / 2.893) | 3.102 (2.892 / 3.309 / 3.104) | 3.170 (3.284 / 3.046 / 3.180) | 3.439 (3.327 / 3.536 / 3.453) | 3.141 (2.588 / 3.427 / 3.408) |
| H3 distance from init, own attention | 3.752 (3.770 / 3.795 / 3.690) | 3.530 (3.341 / 3.719 / 3.529) | 3.621 (3.737 / 3.458 / 3.668) | 3.582 (3.498 / 3.710 / 3.537) | 3.439 (2.939 / 3.711 / 3.667) |
| H3 gradient norm neocortex / own attention | 1.025 (1.160 / 0.867 / 1.046) | 1.771 (2.233 / 1.477 / 1.602) | 1.863 (1.905 / 1.781 / 1.901) | 7.891 (6.798 / 7.725 / 9.151) | 5.060 (5.284 / 3.982 / 5.914) |

## B models: pre-registered verdicts (each T = N neuron vs the memoryless LIF, paired by seed; primary = LIF)

| hypothesis | T = N neuron | part | T = N − memoryless per seed | holds | verdict |
|---|---|---|---|---|---|
| H1 token blurring | B, T = N LIF | own-patch R² lower ≥ 0.02 | -0.079 / -0.079 / -0.070 | yes | partly |
| | | neocortex keys higher ≥ 10 % (rel.) | 5.9 % / -1.3 % / -5.0 % | no | |
| (memory probe) | | last-RR readout higher ≥ 0.02 | 0.044 / 0.047 / 0.045 | yes | |
| H2 position drift | B, T = N LIF | rate / readout first 5 vs steady | rate no, readout no | | not supported |
| H3 less learning | B, T = N LIF | neocortex distance from init lower ≥ 10 % | -4.9 % / 8.0 % / 7.1 % | no | not supported |
| | | gradient neocortex / own lower ≥ 10 % | 92.5 % / 70.3 % / 53.1 % | no | |
| H1 token blurring | B, T = N f-LIF | own-patch R² lower ≥ 0.02 | 0.019 / 0.032 / 0.023 | no | not supported |
| | | neocortex keys higher ≥ 10 % (rel.) | 11.3 % / 1.0 % / -1.3 % | no | |
| (memory probe) | | last-RR readout higher ≥ 0.02 | 0.053 / 0.068 / 0.053 | yes | |
| H2 position drift | B, T = N f-LIF | rate / readout first 5 vs steady | rate no, readout no | | not supported |
| H3 less learning | B, T = N f-LIF | neocortex distance from init lower ≥ 10 % | 7.1 % / -2.9 % / 6.5 % | no | not supported |
| | | gradient neocortex / own lower ≥ 10 % | 64.2 % / 105.4 % / 81.8 % | no | |
| H1 token blurring | B, T = N population LIF | own-patch R² lower ≥ 0.02 | -0.322 / -0.333 / -0.316 | yes | partly |
| | | neocortex keys higher ≥ 10 % (rel.) | 7.1 % / -1.7 % / -9.5 % | no | |
| (memory probe) | | last-RR readout higher ≥ 0.02 | -0.019 / -0.042 / -0.025 | no | |
| H2 position drift | B, T = N population LIF | rate / readout first 5 vs steady | rate no, readout no | | not supported |
| H3 less learning | B, T = N population LIF | neocortex distance from init lower ≥ 10 % | -64.1 % / -63.1 % / -62.8 % | yes | partly |
| | | gradient neocortex / own lower ≥ 10 % | 486.0 % / 790.7 % / 774.7 % | no | |
| H1 token blurring | B, T = N population f-LIF | own-patch R² lower ≥ 0.02 | -0.311 / -0.321 / -0.302 | yes | partly |
| | | neocortex keys higher ≥ 10 % (rel.) | 2.8 % / -2.5 % / -5.4 % | no | |
| (memory probe) | | last-RR readout higher ≥ 0.02 | -0.012 / -0.009 / -0.019 | no | |
| H2 position drift | B, T = N population f-LIF | rate / readout first 5 vs steady | rate no, readout no | | not supported |
| H3 less learning | B, T = N population f-LIF | neocortex distance from init lower ≥ 10 % | -84.4 % / -80.1 % / -79.3 % | yes | partly |
| | | gradient neocortex / own lower ≥ 10 % | 355.5 % / 359.2 % / 465.3 % | no | |
| H4 neocortex unused | LIF and memoryless LIF | swap drop < 0.5 %p (mean) | T = N 11.16 (11.25 / 10.72 / 11.51), memoryless 4.30 (3.34 / 4.98 / 4.57) | | not supported |
| H4 memory content | B, T = N LIF | history-scramble drop < 0.5 %p (mean) | 6.13 (6.74 / 6.12 / 5.55) | | used |
| H4 memory content | B, T = N f-LIF | history-scramble drop < 0.5 %p (mean) | 6.73 (7.38 / 4.67 / 8.13) | | used |
| H4 memory content | B, T = N population LIF | history-scramble drop < 0.5 %p (mean) | 4.34 (4.87 / 4.18 / 3.98) | | used |
| H4 memory content | B, T = N population f-LIF | history-scramble drop < 0.5 %p (mean) | 2.28 (3.06 / -0.10 / 3.87) | | used |

## H5: last k tokens (tiled) and token-block shuffles (test score %, 90-epoch weights)

| model | full | last 4 | last 8 | last 16 | last 32 | last 64 | shuffle b=1 | shuffle b=4 | shuffle b=16 |
|---|---|---|---|---|---|---|---|---|---|
| Spikformer-style | 64.89 (65.65 / 64.98 / 64.06) | 46.81 (46.90 / 48.45 / 45.08) | 53.70 (55.47 / 52.67 / 52.96) | 58.83 (60.25 / 57.85 / 58.38) | 62.77 (62.99 / 62.57 / 62.73) | 64.09 (64.99 / 63.37 / 63.91) | 63.40 (64.25 / 64.08 / 61.88) | 63.78 (64.55 / 64.34 / 62.44) | 64.98 (65.99 / 65.00 / 63.94) |
| Neo-mixer LIF | 62.37 (61.36 / 62.80 / 62.96) | 45.98 (46.29 / 45.95 / 45.72) | 53.93 (55.94 / 53.27 / 52.58) | 59.36 (58.42 / 59.46 / 60.18) | 62.94 (61.43 / 63.69 / 63.71) | 61.90 (59.60 / 62.94 / 63.16) | 53.12 (53.87 / 52.96 / 52.55) | 58.92 (57.50 / 59.56 / 59.69) | 61.89 (60.76 / 61.71 / 63.19) |
| Neo-mixer f-LIF | 61.06 (62.56 / 61.89 / 58.72) | 44.81 (45.59 / 44.44 / 44.42) | 53.10 (54.49 / 53.48 / 51.32) | 58.94 (59.46 / 60.36 / 56.98) | 61.72 (61.44 / 62.94 / 60.77) | 60.59 (61.79 / 60.41 / 59.57) | 47.19 (49.00 / 48.38 / 44.19) | 59.50 (61.58 / 59.68 / 57.24) | 61.63 (61.95 / 62.46 / 60.49) |
| Neo-mixer population LIF | 63.88 (63.37 / 65.12 / 63.16) | 40.99 (41.24 / 42.59 / 39.13) | 49.59 (49.91 / 48.99 / 49.85) | 57.06 (56.85 / 56.69 / 57.63) | 59.89 (59.41 / 59.30 / 60.96) | 62.67 (62.36 / 62.32 / 63.34) | 46.38 (46.92 / 47.41 / 44.81) | 51.56 (52.20 / 51.59 / 50.90) | 61.16 (58.33 / 62.49 / 62.67) |
| Neo-mixer population f-LIF | 61.32 (60.89 / 61.94 / 61.13) | 43.50 (43.63 / 42.21 / 44.67) | 51.02 (50.30 / 52.13 / 50.62) | 56.28 (56.55 / 56.00 / 56.28) | 60.63 (59.91 / 60.79 / 61.17) | 62.25 (61.47 / 61.31 / 63.97) | 49.72 (50.69 / 49.57 / 48.90) | 55.27 (55.08 / 56.55 / 54.18) | 60.86 (60.58 / 61.38 / 60.63) |

- H5 (Spikformer-style): full 64.89, last 16 58.83, shuffle b=1 63.40 → **not supported**
- Neo-mixer LIF: shuffle b=1 drop 9.25 %p vs Spikformer-style 1.49 %p → leans on order more
- Neo-mixer f-LIF: shuffle b=1 drop 13.87 %p vs Spikformer-style 1.49 %p → leans on order more
- Neo-mixer population LIF: shuffle b=1 drop 17.50 %p vs Spikformer-style 1.49 %p → leans on order more
- Neo-mixer population f-LIF: shuffle b=1 drop 11.60 %p vs Spikformer-style 1.49 %p → leans on order more

