# Shuffle-retrain `smoke-rhythm-shuf-20261028` (CINC2017 fold 0, 90 epochs) — summary

Test score (%), best-validation weights; mean (seed 42 / 7 / 2026). Normal order: `flif-v1-rhythm-budget-cinc`. Shuffle-trained models are validated and tested on one fixed token order (default_rng(0).permutation(117)); the eval-shuffle column is the normal-order model tested on that same order (`flif-v1-cause-cinc`).

| model | normal order | shuffle-trained | normal model, eval shuffle | S1 normal − shuffle-trained (%p) | verdict | recovery (%p) |
|---|---|---|---|---|---|---|
| Spikformer-style | 64.89 (65.65 / 64.98 / 64.06) | incomplete | 63.40 (64.25 / 64.08 / 61.88) | incomplete | incomplete | incomplete |
| Neo-mixer LIF | 62.37 (61.36 / 62.80 / 62.96) | incomplete | 53.12 (53.87 / 52.96 / 52.55) | incomplete | incomplete | incomplete |
| Neo-mixer f-LIF | 61.06 (62.56 / 61.89 / 58.72) | incomplete | 47.19 (49.00 / 48.38 / 44.19) | incomplete | incomplete | incomplete |
| Neo-mixer population LIF | 63.88 (63.37 / 65.12 / 63.16) | incomplete | 46.38 (46.92 / 47.41 / 44.81) | incomplete | incomplete | incomplete |
| Neo-mixer population f-LIF | 61.32 (60.89 / 61.94 / 61.13) | incomplete | 49.72 (50.69 / 49.57 / 48.90) | incomplete | incomplete | incomplete |

- S2 CinC rewards order (Spikformer-style S1 = order helps): **incomplete**
- shuffle-trained Neo-mixer LIF − shuffle-trained Spikformer-style (%p): incomplete
- shuffle-trained Neo-mixer f-LIF − shuffle-trained Spikformer-style (%p): incomplete
- shuffle-trained Neo-mixer population LIF − shuffle-trained Spikformer-style (%p): incomplete
- shuffle-trained Neo-mixer population f-LIF − shuffle-trained Spikformer-style (%p): incomplete

