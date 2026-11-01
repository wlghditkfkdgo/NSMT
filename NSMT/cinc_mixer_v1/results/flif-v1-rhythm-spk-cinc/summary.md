# SpkMixer rerun `flif-v1-rhythm-spk-cinc` — verdicts with residual OR

CinC fold 0, test score (mean F1 of N / AF / Other, %), best-validation weights, seeds (42, 7, 2026). Verdict: mean >= +0.5 %p and >= 2/3 seeds = better; mirror = worse; else no clear difference. Spikformer-style reference `spk_ssa`: 65.60 ± 1.21, 127.4 µJ (3 runs, suite `flif-v1-rhythm-cinc`).

## Scores

| neuron | OR fusion | sdor (F v R v s_d) | sdlin (W[F v R; s_d]) | linmx (reference, real input) |
|---|---:|---:|---:|---:|
| LIF | 61.78 ± 0.74 (3) | 63.47 ± 1.13 (3) | 66.96 ± 0.95 (3) | 64.02 ± 0.57 (3) |
| f-LIF | 62.10 ± 0.34 (3) | 62.03 ± 0.45 (3) | 63.73 ± 0.63 (3) | — |
| population LIF (soma) | 62.85 ± 0.45 (3) | 63.48 ± 1.19 (3) | 62.20 ± 0.98 (3) | — |
| population f-LIF (soma) | 62.28 ± 0.15 (3) | 61.62 ± 0.63 (3) | 61.25 ± 1.10 (3) | — |
| popg LIF (tau groups, no soma) | 63.57 ± 1.12 (3) | 64.36 ± 0.69 (3) | 63.55 ± 0.75 (3) | — |
| popg f-LIF (tau groups, no soma) | 60.01 ± 0.44 (3) | 60.81 ± 0.79 (3) | 60.48 ± 1.15 (3) | — |

## Energy (total µJ, FFT neuron model; criterion rule / syops rule)

| neuron | OR | sdor | sdlin | linmx |
|---|---:|---:|---:|---:|
| LIF | 110.5 / 110.5 | 115.9 / 115.9 | 129.8 / 129.8 | 1066.0 / 1066.0 |
| f-LIF | 128.7 / 128.7 | 138.0 / 138.0 | 155.0 / 155.0 | — |
| population LIF (soma) | 199.4 / 198.3 | 217.4 / 216.1 | 244.6 / 243.2 | — |
| population f-LIF (soma) | 332.3 / 331.2 | 378.3 / 377.0 | 432.6 / 431.1 | — |
| popg LIF (tau groups, no soma) | 115.8 / 115.8 | 122.7 / 122.7 | 137.3 / 137.3 | — |
| popg f-LIF (tau groups, no soma) | 132.4 / 132.4 | 143.3 / 143.3 | 160.1 / 160.1 | — |

## V1 fusion effect per neuron (same residual)

| question | contrast | mean %p | per seed | verdict |
|---|---|---:|---|---|
| V1a LIF: s_d by OR | spk_lif_sdor_or − spk_lif_or_or | +1.69 | -0.21, +1.80, +3.49 | better |
| V1b LIF: s_d by learned fusion | spk_lif_sdlin_or − spk_lif_or_or | +5.18 | +3.77, +4.35, +7.42 | better |
| V1c LIF: learned vs OR form | spk_lif_sdlin_or − spk_lif_sdor_or | +3.49 | +3.98, +2.56, +3.93 | better |
| V1a f-LIF: s_d by OR | spk_flif_sdor_or − spk_flif_or_or | -0.07 | -0.94, +0.47, +0.25 | no clear difference |
| V1b f-LIF: s_d by learned fusion | spk_flif_sdlin_or − spk_flif_or_or | +1.63 | +2.18, +1.40, +1.30 | better |
| V1c f-LIF: learned vs OR form | spk_flif_sdlin_or − spk_flif_sdor_or | +1.70 | +3.12, +0.93, +1.05 | better |
| V1a population LIF (soma): s_d by OR | spk_pop_lif_sdor_or − spk_pop_lif_or_or | +0.63 | +2.04, -1.31, +1.16 | better |
| V1b population LIF (soma): s_d by learned fusion | spk_pop_lif_sdlin_or − spk_pop_lif_or_or | -0.65 | -1.88, -1.24, +1.17 | worse |
| V1c population LIF (soma): learned vs OR form | spk_pop_lif_sdlin_or − spk_pop_lif_sdor_or | -1.28 | -3.93, +0.07, +0.02 | no clear difference |
| V1a population f-LIF (soma): s_d by OR | spk_pop_flif_sdor_or − spk_pop_flif_or_or | -0.67 | -0.28, -0.29, -1.44 | worse |
| V1b population f-LIF (soma): s_d by learned fusion | spk_pop_flif_sdlin_or − spk_pop_flif_or_or | -1.03 | +0.55, -2.42, -1.23 | worse |
| V1c population f-LIF (soma): learned vs OR form | spk_pop_flif_sdlin_or − spk_pop_flif_sdor_or | -0.37 | +0.83, -2.13, +0.21 | no clear difference |
| V1a popg LIF (tau groups, no soma): s_d by OR | spk_popg_lif_sdor_or − spk_popg_lif_or_or | +0.79 | +2.14, -0.40, +0.63 | better |
| V1b popg LIF (tau groups, no soma): s_d by learned fusion | spk_popg_lif_sdlin_or − spk_popg_lif_or_or | -0.01 | +1.06, +0.59, -1.69 | no clear difference |
| V1c popg LIF (tau groups, no soma): learned vs OR form | spk_popg_lif_sdlin_or − spk_popg_lif_sdor_or | -0.81 | -1.08, +0.98, -2.32 | worse |
| V1a popg f-LIF (tau groups, no soma): s_d by OR | spk_popg_flif_sdor_or − spk_popg_flif_or_or | +0.80 | +0.24, +0.37, +1.79 | better |
| V1b popg f-LIF (tau groups, no soma): s_d by learned fusion | spk_popg_flif_sdlin_or − spk_popg_flif_or_or | +0.47 | -0.11, +1.49, +0.03 | no clear difference |
| V1c popg f-LIF (tau groups, no soma): learned vs OR form | spk_popg_flif_sdlin_or − spk_popg_flif_sdor_or | -0.33 | -0.34, +1.12, -1.76 | no clear difference |

## V2 vs Spikformer-style (criterion (2): better, or mean >= −0.5 %p with energy <= 0.5x)

| condition | mean %p | per seed | verdict | energy / ref | criterion (2) |
|---|---:|---|---|---:|---|
| spk_lif_or_or | -3.81 | -5.16, -1.92, -4.36 | worse | 0.87x | not met |
| spk_lif_sdor_or | -2.13 | -5.38, -0.13, -0.87 | worse | 0.91x | not met |
| spk_lif_sdlin_or | +1.37 | -1.40, +2.43, +3.06 | better | 1.02x | met |
| spk_lif_linmx_or | -1.57 | -3.97, +0.25, -1.01 | worse | 8.37x | not met |
| spk_flif_or_or | -3.49 | -4.87, -2.79, -2.82 | worse | 1.01x | not met |
| spk_flif_sdor_or | -3.57 | -5.81, -2.31, -2.57 | worse | 1.08x | not met |
| spk_flif_sdlin_or | -1.87 | -2.69, -1.39, -1.53 | worse | 1.22x | not met |
| spk_pop_lif_or_or | -2.74 | -4.33, -1.05, -2.85 | worse | 1.57x | not met |
| spk_pop_lif_sdor_or | -2.11 | -2.29, -2.36, -1.69 | worse | 1.71x | not met |
| spk_pop_lif_sdlin_or | -3.40 | -6.22, -2.30, -1.67 | worse | 1.92x | not met |
| spk_pop_flif_or_or | -3.31 | -5.08, -1.92, -2.94 | worse | 2.61x | not met |
| spk_pop_flif_sdor_or | -3.98 | -5.36, -2.21, -4.38 | worse | 2.97x | not met |
| spk_pop_flif_sdlin_or | -4.35 | -4.53, -4.34, -4.17 | worse | 3.40x | not met |
| spk_popg_lif_or_or | -2.03 | -5.23, -0.40, -0.47 | worse | 0.91x | not met |
| spk_popg_lif_sdor_or | -1.24 | -3.09, -0.80, +0.16 | worse | 0.96x | not met |
| spk_popg_lif_sdlin_or | -2.04 | -4.17, +0.19, -2.15 | worse | 1.08x | not met |
| spk_popg_flif_or_or | -5.59 | -7.73, -3.82, -5.23 | worse | 1.04x | not met |
| spk_popg_flif_sdor_or | -4.79 | -7.49, -3.45, -3.43 | worse | 1.12x | not met |
| spk_popg_flif_sdlin_or | -5.12 | -7.83, -2.33, -5.19 | worse | 1.26x | not met |

## V3 reference (3): LIF linmx (real-valued fusion input)

| question | contrast | mean %p | per seed | verdict |
|---|---|---:|---|---|
| V3a linmx vs sdlin | spk_lif_linmx_or − spk_lif_sdlin_or | -2.94 | -2.57, -2.18, -4.07 | worse |
| V3b linmx vs sdor | spk_lif_linmx_or − spk_lif_sdor_or | +0.55 | +1.41, +0.38, -0.14 | better |

## B1 residual contrasts (LIF, test score; the selection itself used validation)

| fusion | contrast | mean %p | per seed | verdict |
|---|---|---:|---|---|
| or: OR − IAND | spk_lif_or_or − spk_lif_or_iand | -1.30 | -0.43, -0.74, -2.73 | worse |
| or: ADD − IAND | spk_lif_or_add − spk_lif_or_iand | -2.72 | +0.52, -4.72, -3.96 | worse |
| sdor: OR − IAND | spk_lif_sdor_or − spk_lif_sdor_iand | -1.60 | -2.91, +0.03, -1.93 | worse |
| sdor: ADD − IAND | spk_lif_sdor_add − spk_lif_sdor_iand | +0.19 | +0.47, +1.94, -1.85 | no clear difference |
| sdlin: OR − IAND | spk_lif_sdlin_or − spk_lif_sdlin_iand | -0.15 | -2.02, -0.71, +2.27 | no clear difference |
| sdlin: ADD − IAND | spk_lif_sdlin_add − spk_lif_sdlin_iand | -1.21 | -2.35, -1.39, +0.12 | worse |

## All runs

| condition | seed | validation | test score | macro-F1 | AF F1 | Other F1 | Noisy F1 | total µJ syops | total µJ integer | best epoch | time s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spk_flif_or_or | 7 | 66.81 | 61.63 | 59.82 | 54.55 | 48.87 | 54.40 | 128.9 | 128.9 | 26 | 13747 |
| spk_flif_or_or | 42 | 65.95 | 62.38 | 59.10 | 55.52 | 50.46 | 49.23 | 128.6 | 128.6 | 23 | 15165 |
| spk_flif_or_or | 2026 | 66.48 | 62.30 | 58.87 | 52.92 | 52.16 | 48.60 | 128.7 | 128.7 | 24 | 13607 |
| spk_flif_sdlin_or | 7 | 70.22 | 63.03 | 57.18 | 56.16 | 50.64 | 39.62 | 155.0 | 155.0 | 24 | 15058 |
| spk_flif_sdlin_or | 42 | 69.00 | 64.56 | 59.89 | 61.81 | 49.50 | 45.87 | 155.2 | 155.2 | 29 | 17318 |
| spk_flif_sdlin_or | 2026 | 70.64 | 63.59 | 61.92 | 56.20 | 52.03 | 56.91 | 154.8 | 154.8 | 23 | 15632 |
| spk_flif_sdor_or | 7 | 67.65 | 62.10 | 59.18 | 57.23 | 46.76 | 50.43 | 137.8 | 137.8 | 25 | 14184 |
| spk_flif_sdor_or | 42 | 67.36 | 61.44 | 56.63 | 58.86 | 43.76 | 42.20 | 138.0 | 138.0 | 18 | 14628 |
| spk_flif_sdor_or | 2026 | 69.25 | 62.55 | 59.31 | 56.64 | 49.26 | 49.61 | 138.2 | 138.2 | 25 | 13141 |
| spk_lif_linmx_or | 7 | 69.50 | 64.67 | 59.85 | 59.03 | 51.68 | 45.38 | 1065.7 | 1065.7 | 28 | 5395 |
| spk_lif_linmx_or | 42 | 70.31 | 63.28 | 61.10 | 56.18 | 51.38 | 54.55 | 1065.9 | 1065.9 | 24 | 5906 |
| spk_lif_linmx_or | 2026 | 68.70 | 64.11 | 59.71 | 56.74 | 53.02 | 46.51 | 1066.4 | 1066.4 | 25 | 5546 |
| spk_lif_or_add | 7 | 65.78 | 58.51 | 55.37 | 44.73 | 50.12 | 45.93 | 414.4 | 110.4 | 14 | 3718 |
| spk_lif_or_add | 42 | 66.45 | 63.05 | 58.59 | 57.24 | 51.37 | 45.22 | 414.9 | 111.5 | 25 | 7792 |
| spk_lif_or_add | 2026 | 67.36 | 59.53 | 57.56 | 47.29 | 49.94 | 51.67 | 414.7 | 111.3 | 23 | 6566 |
| spk_lif_or_iand | 7 | 66.37 | 63.23 | 58.02 | 56.46 | 50.97 | 42.37 | 119.7 | 119.7 | 26 | 4767 |
| spk_lif_or_iand | 42 | 68.36 | 62.53 | 54.81 | 57.63 | 48.45 | 31.68 | 119.8 | 119.8 | 28 | 7680 |
| spk_lif_or_iand | 2026 | 66.91 | 63.49 | 57.62 | 57.88 | 49.49 | 40.00 | 120.3 | 120.3 | 26 | 4630 |
| spk_lif_or_or | 7 | 66.68 | 62.49 | 57.20 | 56.20 | 51.01 | 41.32 | 110.8 | 110.8 | 23 | 4752 |
| spk_lif_or_or | 42 | 67.49 | 62.09 | 57.87 | 54.48 | 51.69 | 45.22 | 110.0 | 110.0 | 19 | 6342 |
| spk_lif_or_or | 2026 | 68.05 | 60.76 | 56.48 | 51.72 | 48.47 | 43.64 | 110.9 | 110.9 | 26 | 4641 |
| spk_lif_sdlin_add | 7 | 71.05 | 66.17 | 63.07 | 60.65 | 54.13 | 53.78 | 437.2 | 131.6 | 26 | 4895 |
| spk_lif_sdlin_add | 42 | 72.14 | 65.53 | 60.61 | 60.21 | 53.22 | 45.83 | 436.8 | 130.9 | 22 | 4687 |
| spk_lif_sdlin_add | 2026 | 70.82 | 66.03 | 62.34 | 59.65 | 55.04 | 51.28 | 437.4 | 131.9 | 23 | 5858 |
| spk_lif_sdlin_iand | 7 | 70.35 | 67.56 | 61.08 | 63.12 | 55.72 | 41.67 | 143.2 | 143.2 | 23 | 7891 |
| spk_lif_sdlin_iand | 42 | 71.20 | 67.88 | 60.22 | 66.67 | 54.12 | 37.25 | 143.8 | 143.8 | 22 | 6630 |
| spk_lif_sdlin_iand | 2026 | 71.39 | 65.91 | 62.84 | 59.46 | 54.17 | 53.62 | 144.3 | 144.3 | 23 | 6081 |
| spk_lif_sdlin_or | 7 | 71.30 | 66.85 | 61.30 | 61.70 | 55.20 | 44.68 | 129.4 | 129.4 | 23 | 4876 |
| spk_lif_sdlin_or | 42 | 69.89 | 65.86 | 59.97 | 63.27 | 52.05 | 42.31 | 130.1 | 130.1 | 22 | 8585 |
| spk_lif_sdlin_or | 2026 | 74.88 | 68.18 | 63.95 | 65.93 | 56.08 | 51.24 | 129.9 | 129.9 | 23 | 4720 |
| spk_lif_sdor_add | 7 | 70.48 | 66.19 | 62.14 | 62.41 | 53.01 | 50.00 | 420.1 | 117.5 | 28 | 6265 |
| spk_lif_sdor_add | 42 | 68.55 | 65.26 | 60.45 | 61.05 | 51.85 | 46.02 | 420.5 | 117.9 | 26 | 5591 |
| spk_lif_sdor_add | 2026 | 72.21 | 64.34 | 60.34 | 57.86 | 52.88 | 48.33 | 420.0 | 117.7 | 25 | 4125 |
| spk_lif_sdor_iand | 7 | 69.03 | 64.26 | 59.86 | 58.99 | 51.08 | 46.67 | 125.4 | 125.4 | 26 | 6263 |
| spk_lif_sdor_iand | 42 | 69.90 | 64.78 | 59.07 | 58.66 | 52.25 | 41.94 | 126.7 | 126.7 | 26 | 5817 |
| spk_lif_sdor_iand | 2026 | 69.34 | 66.18 | 61.69 | 63.95 | 51.02 | 48.23 | 126.0 | 126.0 | 22 | 5882 |
| spk_lif_sdor_or | 7 | 71.08 | 64.29 | 60.61 | 60.84 | 48.96 | 49.57 | 115.7 | 115.7 | 18 | 6481 |
| spk_lif_sdor_or | 42 | 69.62 | 61.88 | 58.01 | 54.42 | 49.23 | 46.40 | 116.3 | 116.3 | 28 | 6947 |
| spk_lif_sdor_or | 2026 | 70.15 | 64.25 | 60.69 | 56.49 | 54.16 | 50.00 | 115.8 | 115.8 | 23 | 4478 |
| spk_pop_flif_or_or | 7 | 69.42 | 62.49 | 60.29 | 62.67 | 44.98 | 53.66 | 331.4 | 332.6 | 23 | 9306 |
| spk_pop_flif_or_or | 42 | 64.19 | 62.18 | 61.15 | 57.24 | 49.32 | 58.06 | 330.5 | 331.7 | 27 | 11916 |
| spk_pop_flif_or_or | 2026 | 65.74 | 62.18 | 57.28 | 56.76 | 49.00 | 42.59 | 331.6 | 332.7 | 28 | 10525 |
| spk_pop_flif_sdlin_or | 7 | 62.60 | 60.08 | 56.89 | 54.86 | 46.12 | 47.33 | 429.5 | 431.0 | 25 | 17888 |
| spk_pop_flif_sdlin_or | 42 | 62.98 | 62.73 | 58.79 | 57.14 | 49.88 | 46.97 | 429.9 | 431.4 | 24 | 17699 |
| spk_pop_flif_sdlin_or | 2026 | 64.90 | 60.95 | 58.02 | 55.02 | 47.73 | 49.23 | 433.9 | 435.4 | 13 | 19120 |
| spk_pop_flif_sdor_or | 7 | 64.18 | 62.21 | 58.86 | 56.83 | 49.88 | 48.82 | 377.0 | 378.3 | 25 | 10109 |
| spk_pop_flif_sdor_or | 42 | 65.37 | 61.90 | 57.96 | 56.54 | 49.83 | 46.15 | 376.4 | 377.7 | 28 | 9523 |
| spk_pop_flif_sdor_or | 2026 | 66.50 | 60.74 | 58.97 | 55.78 | 46.70 | 53.66 | 377.5 | 378.8 | 28 | 14469 |
| spk_pop_lif_or_or | 7 | 68.76 | 63.36 | 61.20 | 62.02 | 48.48 | 54.70 | 198.0 | 199.1 | 23 | 15229 |
| spk_pop_lif_or_or | 42 | 69.09 | 62.92 | 59.46 | 60.90 | 47.64 | 49.09 | 197.8 | 198.9 | 26 | 10757 |
| spk_pop_lif_or_or | 2026 | 66.37 | 62.27 | 59.10 | 60.93 | 45.69 | 49.60 | 199.1 | 200.2 | 20 | 16684 |
| spk_pop_lif_sdlin_or | 7 | 67.34 | 62.12 | 59.51 | 59.34 | 48.03 | 51.67 | 240.8 | 242.3 | 27 | 22812 |
| spk_pop_lif_sdlin_or | 42 | 65.82 | 61.04 | 56.79 | 56.64 | 47.24 | 44.03 | 247.3 | 248.7 | 13 | 21773 |
| spk_pop_lif_sdlin_or | 2026 | 66.66 | 63.45 | 58.98 | 62.63 | 47.62 | 45.59 | 241.4 | 242.9 | 16 | 25298 |
| spk_pop_lif_sdor_or | 7 | 67.44 | 62.05 | 59.42 | 57.04 | 49.44 | 51.52 | 215.1 | 216.4 | 29 | 18430 |
| spk_pop_lif_sdor_or | 42 | 66.94 | 64.97 | 60.13 | 62.45 | 51.53 | 45.61 | 216.3 | 217.6 | 27 | 16033 |
| spk_pop_lif_sdor_or | 2026 | 66.70 | 63.43 | 61.25 | 59.15 | 51.00 | 54.72 | 216.9 | 218.2 | 28 | 21781 |
| spk_popg_flif_or_or | 7 | 66.31 | 60.59 | 58.23 | 54.43 | 47.28 | 51.13 | 131.6 | 131.6 | 23 | 33567 |
| spk_popg_flif_or_or | 42 | 64.70 | 59.53 | 55.81 | 50.88 | 46.96 | 44.64 | 133.0 | 133.0 | 21 | 29297 |
| spk_popg_flif_or_or | 2026 | 63.91 | 59.89 | 58.36 | 53.68 | 45.51 | 53.75 | 132.4 | 132.4 | 19 | 31671 |
| spk_popg_flif_sdlin_or | 7 | 63.34 | 62.08 | 58.56 | 59.29 | 47.09 | 48.00 | 160.0 | 160.0 | 27 | 40425 |
| spk_popg_flif_sdlin_or | 42 | 62.03 | 59.42 | 56.98 | 52.23 | 46.56 | 49.66 | 161.2 | 161.2 | 23 | 38742 |
| spk_popg_flif_sdlin_or | 2026 | 63.84 | 59.93 | 56.55 | 53.10 | 47.49 | 46.43 | 158.9 | 158.9 | 28 | 34443 |
| spk_popg_flif_sdor_or | 7 | 65.34 | 60.96 | 57.68 | 57.95 | 46.88 | 47.83 | 142.5 | 142.5 | 23 | 35109 |
| spk_popg_flif_sdor_or | 42 | 62.96 | 59.77 | 56.91 | 53.00 | 47.54 | 48.33 | 143.6 | 143.6 | 19 | 32606 |
| spk_popg_flif_sdor_or | 2026 | 64.53 | 61.69 | 57.47 | 53.12 | 51.84 | 44.83 | 143.8 | 143.8 | 27 | 32562 |
| spk_popg_lif_or_or | 7 | 72.66 | 64.02 | 60.03 | 61.54 | 50.28 | 48.06 | 115.7 | 115.7 | 23 | 10758 |
| spk_popg_lif_or_or | 42 | 66.52 | 62.03 | 58.09 | 58.09 | 49.23 | 46.28 | 115.9 | 115.9 | 22 | 11982 |
| spk_popg_lif_or_or | 2026 | 66.95 | 64.65 | 60.99 | 62.32 | 50.88 | 50.00 | 115.8 | 115.8 | 26 | 7152 |
| spk_popg_lif_sdlin_or | 7 | 66.20 | 64.60 | 60.45 | 64.86 | 48.69 | 48.00 | 136.7 | 136.7 | 24 | 10954 |
| spk_popg_lif_sdlin_or | 42 | 66.17 | 63.08 | 59.73 | 60.07 | 49.30 | 49.66 | 136.9 | 136.9 | 18 | 13204 |
| spk_popg_lif_sdlin_or | 2026 | 66.19 | 62.97 | 59.24 | 58.78 | 48.51 | 48.06 | 138.3 | 138.3 | 19 | 16175 |
| spk_popg_lif_sdor_or | 7 | 68.28 | 63.62 | 61.13 | 60.87 | 49.18 | 53.66 | 122.3 | 122.3 | 25 | 11035 |
| spk_popg_lif_sdor_or | 42 | 67.64 | 64.17 | 58.68 | 62.55 | 49.88 | 42.20 | 122.6 | 122.6 | 20 | 11607 |
| spk_popg_lif_sdor_or | 2026 | 67.51 | 65.28 | 61.66 | 65.45 | 49.76 | 50.79 | 123.4 | 123.4 | 29 | 10015 |

RESULT lines: 75; SKIP lines: 0.
