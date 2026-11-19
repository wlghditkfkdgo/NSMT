# MIT-BIH: test-time token-order shuffle of the stored models (post hoc, not pre-registered, no training)

See the docstring of order_eval_nope.py. macro-F1 %, mean of seeds 42 / 7 / 2026 (per-seed drops in brackets); "changed" = share of the 32833 test samples whose prediction changes. 11 tokens.

| model | bias | normal order | all tokens shuffled | drop %p (per seed) | changed % | Neocortex order only shuffled | drop %p (per seed) | changed % |
|---|---|---|---|---|---|---|---|---|
| Spikformer(DE) | on | 91.94 | 72.04 | -19.91 (-15.90 / -21.74 / -22.08) | 10.53 | — | — | — |
| Spikformer(DE) | off | 91.27 | 91.27 | +0.00 (+0.00 / +0.00 / +0.00) | 0.00 | — | — | — |
| base pop LIF | on | 91.90 | 74.28 | -17.62 (-17.59 / -19.57 / -15.71) | 7.27 | 88.22 | -3.69 (-3.28 / -2.00 / -5.77) | 2.05 |
| base pop LIF | off | 91.51 | 82.96 | -8.54 (-8.53 / -9.79 / -7.31) | 3.90 | 82.96 | -8.54 (-8.53 / -9.79 / -7.31) | 3.90 |
| base pop f-LIF | on | 92.01 | 76.19 | -15.82 (-12.20 / -14.91 / -20.34) | 5.31 | 87.71 | -4.30 (-2.05 / -1.77 / -9.08) | 1.56 |
| base pop f-LIF | off | 91.41 | 85.32 | -6.09 (-6.07 / -7.72 / -4.48) | 2.53 | 85.32 | -6.09 (-6.07 / -7.72 / -4.48) | 2.53 |
