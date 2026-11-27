# Sleep-EDF: adoption-bar reading (report only)

Bar (user 11-27): mean macro-F1 >= +0.1 %p above Spikformer(DE) and syops energy <= 1.5x. Spikformer(DE): lr 0.001 73.06 (16.14 µJ), lr 0.003 74.03 (16.12 µJ); stronger on test = lr 0.003.

| model | lr | macro-F1 | vs Spikformer(DE) same lr (positive folds) | energy × | adoption (same lr) | vs stronger Spikformer(DE) | adoption (conservative) |
|---|---|---|---|---|---|---|---|
| q_lif_acc | 0.003 | 75.41 | +1.38 (3/5) | 1.22 | **pass** | +1.38 | pass |
| q_lif_mem_acc | 0.003 | 74.65 | +0.62 (4/5) | 1.73 | **energy over** | +0.62 | energy over |
| q_pop_lif_mem | 0.003 | 74.63 | +0.60 (3/5) | 1.73 | **energy over** | +0.60 | energy over |
| q_pop_lif | 0.001 | 74.62 | +1.56 (5/5) | 1.13 | **pass** | +0.59 | pass |
| q_flif | 0.003 | 74.40 | +0.37 (2/5) | 1.19 | **pass** | +0.37 | pass |
| q_pop_lif_mem_acc | 0.003 | 74.17 | +0.14 (3/5) | 1.73 | **energy over** | +0.14 | energy over |
| q_pop_lif | 0.003 | 74.16 | +0.13 (4/5) | 1.15 | **pass** | +0.13 | pass |
| q_flif_mem | 0.003 | 74.04 | +0.01 (3/5) | 1.72 | **below** | +0.01 | below |
| q_lif | 0.001 | 74.00 | +0.93 (5/5) | 1.20 | **pass** | -0.03 | below |
| q_pop_lif_acc | 0.003 | 73.87 | -0.16 (3/5) | 1.20 | **below** | -0.16 | below |
| kv_lif | 0.003 | 73.70 | -0.33 (2/5) | 1.16 | **below** | -0.33 | below |
| kv_lif | 0.001 | 73.48 | +0.41 (5/5) | 1.16 | **pass** | -0.55 | below |
| q_lif_mem | 0.003 | 73.45 | -0.58 (2/5) | 1.73 | **below** | -0.58 | below |
| q_lif | 0.003 | 73.39 | -0.64 (1/5) | 1.21 | **below** | -0.64 | below |
| kv_pop_lif | 0.003 | 72.33 | -1.70 (1/5) | 1.04 | **below** | -1.70 | below |

Small gains sit inside the fold-to-fold spread (one seed); the pre-registered clear-difference verdicts are in summary.md.
