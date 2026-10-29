# SpkMixer rerun `flif-v1-rhythm-spk-cinc` — verdicts with residual OR

CinC fold 0, test score (mean F1 of N / AF / Other, %), best-validation weights, seeds (42, 7, 2026). Verdict: mean >= +0.5 %p and >= 2/3 seeds = better; mirror = worse; else no clear difference. Spikformer-style reference `spk_ssa`: 65.60 ± 1.21, 127.4 µJ (3 runs, suite `flif-v1-rhythm-cinc`).

## Scores

| neuron | OR fusion | sdor (F v R v s_d) | sdlin (W[F v R; s_d]) | linmx (reference, real input) |
|---|---:|---:|---:|---:|
| LIF | 61.78 ± 0.74 (3) | 63.47 ± 1.13 (3) | 66.96 ± 0.95 (3) | — |
| f-LIF | — | — | — | — |
| population LIF (soma) | — | — | — | — |
| population f-LIF (soma) | — | — | — | — |
| popg LIF (tau groups, no soma) | — | — | — | — |
| popg f-LIF (tau groups, no soma) | — | — | — | — |

## Energy (total µJ, FFT neuron model; criterion rule / syops rule)

| neuron | OR | sdor | sdlin | linmx |
|---|---:|---:|---:|---:|
| LIF | 110.5 / 110.5 | 115.9 / 115.9 | 129.8 / 129.8 | — |
| f-LIF | — | — | — | — |
| population LIF (soma) | — | — | — | — |
| population f-LIF (soma) | — | — | — | — |
| popg LIF (tau groups, no soma) | — | — | — | — |
| popg f-LIF (tau groups, no soma) | — | — | — | — |

## V1 fusion effect per neuron (same residual)

| question | contrast | mean %p | per seed | verdict |
|---|---|---:|---|---|
| V1a LIF: s_d by OR | spk_lif_sdor_or − spk_lif_or_or | +1.69 | -0.21, +1.80, +3.49 | better |
| V1b LIF: s_d by learned fusion | spk_lif_sdlin_or − spk_lif_or_or | +5.18 | +3.77, +4.35, +7.42 | better |
| V1c LIF: learned vs OR form | spk_lif_sdlin_or − spk_lif_sdor_or | +3.49 | +3.98, +2.56, +3.93 | better |
| V1a f-LIF: s_d by OR | spk_flif_sdor_or − spk_flif_or_or | incomplete | | incomplete |
| V1b f-LIF: s_d by learned fusion | spk_flif_sdlin_or − spk_flif_or_or | incomplete | | incomplete |
| V1c f-LIF: learned vs OR form | spk_flif_sdlin_or − spk_flif_sdor_or | incomplete | | incomplete |
| V1a population LIF (soma): s_d by OR | spk_pop_lif_sdor_or − spk_pop_lif_or_or | incomplete | | incomplete |
| V1b population LIF (soma): s_d by learned fusion | spk_pop_lif_sdlin_or − spk_pop_lif_or_or | incomplete | | incomplete |
| V1c population LIF (soma): learned vs OR form | spk_pop_lif_sdlin_or − spk_pop_lif_sdor_or | incomplete | | incomplete |
| V1a population f-LIF (soma): s_d by OR | spk_pop_flif_sdor_or − spk_pop_flif_or_or | incomplete | | incomplete |
| V1b population f-LIF (soma): s_d by learned fusion | spk_pop_flif_sdlin_or − spk_pop_flif_or_or | incomplete | | incomplete |
| V1c population f-LIF (soma): learned vs OR form | spk_pop_flif_sdlin_or − spk_pop_flif_sdor_or | incomplete | | incomplete |
| V1a popg LIF (tau groups, no soma): s_d by OR | spk_popg_lif_sdor_or − spk_popg_lif_or_or | incomplete | | incomplete |
| V1b popg LIF (tau groups, no soma): s_d by learned fusion | spk_popg_lif_sdlin_or − spk_popg_lif_or_or | incomplete | | incomplete |
| V1c popg LIF (tau groups, no soma): learned vs OR form | spk_popg_lif_sdlin_or − spk_popg_lif_sdor_or | incomplete | | incomplete |
| V1a popg f-LIF (tau groups, no soma): s_d by OR | spk_popg_flif_sdor_or − spk_popg_flif_or_or | incomplete | | incomplete |
| V1b popg f-LIF (tau groups, no soma): s_d by learned fusion | spk_popg_flif_sdlin_or − spk_popg_flif_or_or | incomplete | | incomplete |
| V1c popg f-LIF (tau groups, no soma): learned vs OR form | spk_popg_flif_sdlin_or − spk_popg_flif_sdor_or | incomplete | | incomplete |

## V2 vs Spikformer-style (criterion (2): better, or mean >= −0.5 %p with energy <= 0.5x)

| condition | mean %p | per seed | verdict | energy / ref | criterion (2) |
|---|---:|---|---|---:|---|
| spk_lif_or_or | -3.81 | -5.16, -1.92, -4.36 | worse | 0.87x | not met |
| spk_lif_sdor_or | -2.13 | -5.38, -0.13, -0.87 | worse | 0.91x | not met |
| spk_lif_sdlin_or | +1.37 | -1.40, +2.43, +3.06 | better | 1.02x | met |

## V3 reference (3): LIF linmx (real-valued fusion input)

| question | contrast | mean %p | per seed | verdict |
|---|---|---:|---|---|
| V3a linmx vs sdlin | spk_lif_linmx_or − spk_lif_sdlin_or | incomplete | | incomplete |
| V3b linmx vs sdor | spk_lif_linmx_or − spk_lif_sdor_or | incomplete | | incomplete |

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

RESULT lines: 27; SKIP lines: 0.
