# Shuffle-retrain `flif-v1-rhythm-shuf-cinc` (CINC2017 fold 0, 90 epochs) — summary

Test score (%), best-validation weights; mean (seed 42 / 7 / 2026). Normal order: `flif-v1-rhythm-budget-cinc`. Shuffle-trained models are validated and tested on one fixed token order (default_rng(0).permutation(117)); the eval-shuffle column is the normal-order model tested on that same order (`flif-v1-cause-cinc`).

| model | normal order | shuffle-trained | normal model, eval shuffle | S1 normal − shuffle-trained (%p) | verdict | recovery (%p) |
|---|---|---|---|---|---|---|
| Spikformer-style | 64.89 (65.65 / 64.98 / 64.06) | 64.96 (64.25 / 64.91 / 65.72) | 63.40 (64.25 / 64.08 / 61.88) | -0.06 (1.40 / 0.07 / -1.66) | no clear difference | 1.55 (-0.01 / 0.83 / 3.83) |
| Neo-mixer LIF | 62.37 (61.36 / 62.80 / 62.96) | 59.79 (58.79 / 59.85 / 60.72) | 53.12 (53.87 / 52.96 / 52.55) | 2.59 (2.57 / 2.95 / 2.24) | order helps | 6.67 (4.93 / 6.89 / 8.18) |
| Neo-mixer f-LIF | 61.06 (62.56 / 61.89 / 58.72) | 60.35 (59.66 / 61.55 / 59.85) | 47.19 (49.00 / 48.38 / 44.19) | 0.71 (2.90 / 0.35 / -1.13) | order helps | 13.16 (10.66 / 13.17 / 15.66) |
| Neo-mixer population LIF | 63.88 (63.37 / 65.12 / 63.16) | 60.15 (59.88 / 59.53 / 61.04) | 46.38 (46.92 / 47.41 / 44.81) | 3.74 (3.49 / 5.60 / 2.12) | order helps | 13.77 (12.95 / 12.12 / 16.23) |
| Neo-mixer population f-LIF | 61.32 (60.89 / 61.94 / 61.13) | 58.78 (60.02 / 58.65 / 57.67) | 49.72 (50.69 / 49.57 / 48.90) | 2.53 (0.86 / 3.29 / 3.45) | order helps | 9.06 (9.33 / 9.08 / 8.77) |

- S2 CinC rewards order (Spikformer-style S1 = order helps): **no**
- shuffle-trained Neo-mixer LIF − shuffle-trained Spikformer-style (%p): -5.17 (-5.45 / -5.06 / -4.99)
- shuffle-trained Neo-mixer f-LIF − shuffle-trained Spikformer-style (%p): -4.60 (-4.59 / -3.36 / -5.87)
- shuffle-trained Neo-mixer population LIF − shuffle-trained Spikformer-style (%p): -4.81 (-4.37 / -5.38 / -4.68)
- shuffle-trained Neo-mixer population f-LIF − shuffle-trained Spikformer-style (%p): -6.17 (-4.22 / -6.26 / -8.04)

