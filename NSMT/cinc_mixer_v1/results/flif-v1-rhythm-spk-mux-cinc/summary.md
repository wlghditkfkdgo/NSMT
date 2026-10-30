# Fusion MUX `flif-v1-rhythm-spk-mux-cinc` — verdicts (LIF, residual OR, CinC fold 0)

Test score (mean F1 of N / AF / Other, %), best-validation weights, seeds (42, 7, 2026). Verdict: mean >= +0.5 %p and >= 2/3 seeds = better; mirror = worse; else no clear difference. Spikformer-style `spk_ssa`: 65.60 ± 1.21, 127.4 µJ. Controls from `flif-v1-rhythm-spk-cinc` (B1).

## Scores

| fusion | n | validation | test score | macro-F1 | N F1 | AF F1 | Other F1 | Noisy F1 | total µJ (criterion) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| or: F v R | 3 | 67.41 ± 0.56 | 61.78 ± 0.74 | 57.18 ± 0.57 | 80.82 ± 0.91 | 54.14 ± 1.85 | 50.39 ± 1.39 | 43.39 ± 1.60 | 110.5 |
| sdor: F v R v s_d | 3 | 70.28 ± 0.60 | 63.47 ± 1.13 | 59.77 ± 1.24 | 82.38 ± 0.48 | 57.25 ± 2.68 | 50.79 ± 2.39 | 48.66 ± 1.61 | 115.9 |
| sdlin: L | 3 | 72.02 ± 2.10 | 66.96 ± 0.95 | 61.74 ± 1.65 | 82.80 ± 0.60 | 63.64 ± 1.75 | 54.45 ± 1.73 | 46.08 ± 3.78 | 129.8 |
| (i) muxor: s_d ? F v R : F ^ R | 3 | 69.57 ± 0.24 | 65.21 ± 0.63 | 61.01 ± 0.65 | 82.68 ± 0.43 | 59.60 ± 1.98 | 53.34 ± 0.82 | 48.44 ± 1.83 | 111.1 |
| (ii) muxlin: s_d ? L : F ^ R | 3 | 72.46 ± 1.10 | 68.52 ± 0.81 | 64.35 ± 0.68 | 83.51 ± 0.74 | 65.16 ± 1.46 | 56.88 ± 0.37 | 51.85 ± 1.03 | 129.4 |
| Spikformer-style | 3 | — | 65.60 ± 1.21 | 61.94 ± 0.34 | 82.31 ± 0.62 | 59.78 ± 2.53 | 54.70 ± 0.86 | 50.97 ± 3.82 | 127.4 |

## Pre-registered contrasts

| question | contrast | mean %p | per seed (42, 7, 2026) | verdict |
|---|---|---:|---|---|
| X1 (i) majority vs OR of F, R, s_d | spk_lif_muxor_or − spk_lif_sdor_or | +1.73 | +4.20, +0.64, +0.36 | better |
| X2 (ii) MUX around the learned fusion | spk_lif_muxlin_or − spk_lif_sdlin_or | +1.56 | +3.64, +0.67, +0.37 | better |
| X3a (i) vs spike OR fusion | spk_lif_muxor_or − spk_lif_or_or | +3.42 | +3.99, +2.43, +3.85 | better |
| X3b (ii) vs (i) | spk_lif_muxlin_or − spk_lif_muxor_or | +3.31 | +3.42, +2.59, +3.94 | better |

## X4 vs Spikformer-style (criterion (2))

| condition | mean %p | per seed | verdict | energy / ref | criterion (2) |
|---|---:|---|---|---:|---|
| spk_lif_muxor_or | -0.39 | -1.18, +0.51, -0.51 | no clear difference | 0.87x | not met |
| spk_lif_muxlin_or | +2.92 | +2.24, +3.10, +3.43 | better | 1.02x | met |

## New runs

| condition | seed | validation | test score | macro-F1 | AF F1 | Other F1 | Noisy F1 | total µJ | best epoch | time s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spk_lif_muxlin_or | 7 | 70.96 | 67.51 | 63.74 | 63.64 | 56.43 | 52.43 | 129.0 | 16 | 9135 |
| spk_lif_muxlin_or | 42 | 72.84 | 69.50 | 65.30 | 67.12 | 57.35 | 52.71 | 129.7 | 23 | 10152 |
| spk_lif_muxlin_or | 2026 | 73.58 | 68.55 | 64.01 | 64.73 | 56.87 | 50.41 | 129.5 | 29 | 10822 |
| spk_lif_muxor_or | 7 | 69.31 | 64.93 | 61.42 | 59.28 | 52.37 | 50.91 | 111.2 | 22 | 7572 |
| spk_lif_muxor_or | 42 | 69.89 | 66.08 | 61.53 | 62.16 | 53.25 | 47.86 | 111.3 | 28 | 7897 |
| spk_lif_muxor_or | 2026 | 69.50 | 64.61 | 60.10 | 57.35 | 54.38 | 46.55 | 110.8 | 23 | 7075 |

RESULT lines (new): 6; SKIP lines: 0.
