# Fusion MUX `smoke-rhythm-spk-mux-20261030` — verdicts (LIF, residual OR, CinC fold 0)

Test score (mean F1 of N / AF / Other, %), best-validation weights, seeds (42, 7, 2026). Verdict: mean >= +0.5 %p and >= 2/3 seeds = better; mirror = worse; else no clear difference. Spikformer-style `spk_ssa`: 65.60 ± 1.21, 127.4 µJ. Controls from `flif-v1-rhythm-spk-cinc` (B1).

## Scores

| fusion | n | validation | test score | macro-F1 | N F1 | AF F1 | Other F1 | Noisy F1 | total µJ (criterion) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| or: F v R | 3 | 67.41 ± 0.56 | 61.78 ± 0.74 | 57.18 ± 0.57 | 80.82 ± 0.91 | 54.14 ± 1.85 | 50.39 ± 1.39 | 43.39 ± 1.60 | 110.5 |
| sdor: F v R v s_d | 3 | 70.28 ± 0.60 | 63.47 ± 1.13 | 59.77 ± 1.24 | 82.38 ± 0.48 | 57.25 ± 2.68 | 50.79 ± 2.39 | 48.66 ± 1.61 | 115.9 |
| sdlin: L | 3 | 72.02 ± 2.10 | 66.96 ± 0.95 | 61.74 ± 1.65 | 82.80 ± 0.60 | 63.64 ± 1.75 | 54.45 ± 1.73 | 46.08 ± 3.78 | 129.8 |
| (i) muxor: s_d ? F v R : F ^ R | 1 | 44.06 | 44.27 | 42.09 | 77.75 | 19.10 | 35.96 | 35.56 | 109.6 |
| (ii) muxlin: s_d ? L : F ^ R | 1 | 39.61 | 38.18 | 37.16 | 77.28 | 10.84 | 26.41 | 34.11 | 128.4 |
| Spikformer-style | 3 | — | 65.60 ± 1.21 | 61.94 ± 0.34 | 82.31 ± 0.62 | 59.78 ± 2.53 | 54.70 ± 0.86 | 50.97 ± 3.82 | 127.4 |

## Pre-registered contrasts

| question | contrast | mean %p | per seed (42, 7, 2026) | verdict |
|---|---|---:|---|---|
| X1 (i) majority vs OR of F, R, s_d | spk_lif_muxor_or − spk_lif_sdor_or | incomplete | | incomplete |
| X2 (ii) MUX around the learned fusion | spk_lif_muxlin_or − spk_lif_sdlin_or | incomplete | | incomplete |
| X3a (i) vs spike OR fusion | spk_lif_muxor_or − spk_lif_or_or | incomplete | | incomplete |
| X3b (ii) vs (i) | spk_lif_muxlin_or − spk_lif_muxor_or | incomplete | | incomplete |

## X4 vs Spikformer-style (criterion (2))

| condition | mean %p | per seed | verdict | energy / ref | criterion (2) |
|---|---:|---|---|---:|---|
| spk_lif_muxor_or | incomplete | | incomplete | | incomplete |
| spk_lif_muxlin_or | incomplete | | incomplete | | incomplete |

## New runs

| condition | seed | validation | test score | macro-F1 | AF F1 | Other F1 | Noisy F1 | total µJ | best epoch | time s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| spk_lif_muxlin_or | 42 | 39.61 | 38.18 | 37.16 | 10.84 | 26.41 | 34.11 | 128.4 | 0 | 631 |
| spk_lif_muxor_or | 42 | 44.06 | 44.27 | 42.09 | 19.10 | 35.96 | 35.56 | 109.6 | 0 | 445 |

RESULT lines (new): 2; SKIP lines: 0.
