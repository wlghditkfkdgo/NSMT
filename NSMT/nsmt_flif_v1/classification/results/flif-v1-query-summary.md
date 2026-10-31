# Query batch (Neocortex as the attention query) — summary

See the module docstring for names, sources and rules. Exploratory (one CinC fold, 3 seeds).

## CinC 2017 fold 0, 90 epochs (test score %, best-validation weights)

| question | A | B | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|---|---|
| Q1 query vs replace, LIF (main) | nsmt4q_lif | nsmt4_lif | 65.30 | 62.97 | +2.70 / +1.82 / +2.48 | +2.33 | better | 1.02 |  |
| Q1 query vs replace, f-LIF (main) | nsmt4q_flif | nsmt4_flif | 65.25 | 60.26 | +4.19 / +5.85 / +4.93 | +4.99 | better | 1.05 |  |
| Q1 query vs replace, pop LIF (main) | nsmt4q_pop_lif | nsmt4_pop_lif | 66.60 | 63.51 | +3.99 / +2.18 / +3.12 | +3.10 | better | 1.07 |  |
| Q1 query vs replace, pop f-LIF (main) | nsmt4q_pop_flif | nsmt4_pop_flif | 66.47 | 60.63 | +6.65 / +5.63 / +5.24 | +5.84 | better | 1.07 |  |
| Q1 (report) query vs replace, memoryless LIF | nsmt4q_lif_nomem | nsmt4_lif_nomem | 64.94 | 59.12 | +5.46 / +5.57 / +6.44 | +5.82 | better | 1.03 |  |
| Q2 query vs append (B form), LIF | nsmt4q_lif | spkneo_lif | 65.30 | 65.49 | -0.42 / -0.57 / +0.41 | -0.19 | no clear difference | 0.86 |  |
| Q2 query vs append (B form), f-LIF | nsmt4q_flif | spkneo_flif | 65.25 | 64.25 | -0.64 / +1.52 / +2.11 | +1.00 | better | 0.86 |  |
| Q2 query vs append (B form), pop LIF | nsmt4q_pop_lif | spkneo_pop_lif | 66.60 | 65.38 | -0.15 / +0.34 / +3.49 | +1.23 | better | 0.89 |  |
| Q2 query vs append (B form), pop f-LIF | nsmt4q_pop_flif | spkneo_pop_flif | 66.47 | 65.28 | +3.08 / +1.30 / -0.81 | +1.19 | better | 0.91 |  |
| Q3 query vs Spikformer-style, LIF (criterion 2) | nsmt4q_lif | spk_ssa | 65.30 | 64.89 | -0.18 / +0.03 / +1.37 | +0.41 | no clear difference | 1.13 | not met |
| Q3 query vs Spikformer-style, f-LIF (criterion 2) | nsmt4q_flif | spk_ssa | 65.25 | 64.89 | -0.63 / -0.75 / +2.43 | +0.35 | no clear difference | 1.15 | not met |
| Q3 query vs Spikformer-style, pop LIF (criterion 2) | nsmt4q_pop_lif | spk_ssa | 66.60 | 64.89 | +0.22 / +1.53 / +3.38 | +1.71 | better | 1.21 | met |
| Q3 query vs Spikformer-style, pop f-LIF (criterion 2) | nsmt4q_pop_flif | spk_ssa | 66.47 | 64.89 | +1.25 / +1.27 / +2.20 | +1.57 | better | 1.38 | met |
| Q4 T = N memory vs memoryless, query LIF (main) | nsmt4q_lif | nsmt4q_lif_nomem | 65.30 | 64.94 | +1.22 / -0.46 / +0.32 | +0.36 | no clear difference | 1.03 |  |
| Q4 (report) f-LIF vs memoryless LIF, query | nsmt4q_flif | nsmt4q_lif_nomem | 65.25 | 64.94 | +0.78 / -1.24 / +1.38 | +0.31 | no clear difference | 1.05 |  |
| Q4 (report) pop LIF vs memoryless LIF, query | nsmt4q_pop_lif | nsmt4q_lif_nomem | 66.60 | 64.94 | +1.62 / +1.03 / +2.33 | +1.66 | better | 1.11 |  |
| Q4 (report) pop f-LIF vs memoryless LIF, query | nsmt4q_pop_flif | nsmt4q_lif_nomem | 66.47 | 64.94 | +2.65 / +0.78 / +1.15 | +1.53 | better | 1.25 |  |
| Q5 (report) f-LIF vs LIF, query | nsmt4q_flif | nsmt4q_lif | 65.25 | 65.30 | -0.45 / -0.78 / +1.07 | -0.05 | no clear difference | 1.02 |  |
| Q5 (report) pop LIF vs LIF, query | nsmt4q_pop_lif | nsmt4q_lif | 66.60 | 65.30 | +0.39 / +1.50 / +2.01 | +1.30 | better | 1.08 |  |
| Q5 (report) pop f-LIF vs LIF, query | nsmt4q_pop_flif | nsmt4q_lif | 66.47 | 65.30 | +1.43 / +1.24 / +0.84 | +1.17 | better | 1.22 |  |

## ETT forecasting, ETTh2 x H96 / H720 (test MSE; r = 100 (A − B) / B, %)

| question | A | B | cells b / . / w | R | pairs A < B | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|---|
| Q1 query vs replace, LIF (main) | nsmt4q_lif | nsmt4_lif | 0 / 2 / 0 | -0.13 | 4/6 | no clear difference | 1.00 |  |
| Q1 query vs replace, f-LIF (main) | nsmt4q_flif | nsmt4_flif | 0 / 2 / 0 | +0.33 | 2/6 | no clear difference | 1.00 |  |
| Q1 query vs replace, pop LIF (main) | nsmt4q_pop_lif | nsmt4_pop_lif | 0 / 2 / 0 | +0.45 | 1/6 | no clear difference | 1.00 |  |
| Q1 query vs replace, pop f-LIF (main) | nsmt4q_pop_flif | nsmt4_pop_flif | 0 / 2 / 0 | -0.83 | 4/6 | no clear difference | 1.01 |  |
| Q1 (report) query vs replace, memoryless LIF | nsmt4q_lif_nomem | nsmt4_lif_nomem | 0 / 2 / 0 | +0.13 | 2/6 | no clear difference | 1.01 |  |
| Q2 query vs append (B form), LIF | nsmt4q_lif | spkneo_lif | 0 / 2 / 0 | +0.53 | 3/6 | no clear difference | 0.91 |  |
| Q2 query vs append (B form), f-LIF | nsmt4q_flif | spkneo_flif | 0 / 1 / 1 | +0.98 | 2/6 | no clear difference | 0.91 |  |
| Q2 query vs append (B form), pop LIF | nsmt4q_pop_lif | spkneo_pop_lif | 1 / 1 / 0 | -1.13 | 4/6 | better | 0.91 |  |
| Q2 query vs append (B form), pop f-LIF | nsmt4q_pop_flif | spkneo_pop_flif | 1 / 1 / 0 | -0.79 | 4/6 | no clear difference | 0.93 |  |
| Q3 query vs Spikformer-style, LIF (criterion 2) | nsmt4q_lif | spk_ssa | 1 / 1 / 0 | -0.17 | 3/6 | no clear difference | 1.04 | not met |
| Q3 query vs Spikformer-style, f-LIF (criterion 2) | nsmt4q_flif | spk_ssa | 1 / 1 / 0 | -0.22 | 4/6 | no clear difference | 1.05 | not met |
| Q3 query vs Spikformer-style, pop LIF (criterion 2) | nsmt4q_pop_lif | spk_ssa | 1 / 1 / 0 | -0.89 | 4/6 | no clear difference | 1.09 | not met |
| Q3 query vs Spikformer-style, pop f-LIF (criterion 2) | nsmt4q_pop_flif | spk_ssa | 1 / 1 / 0 | -1.26 | 4/6 | better | 1.20 | met |
| Q4 T = N memory vs memoryless, query LIF (main) | nsmt4q_lif | nsmt4q_lif_nomem | 0 / 2 / 0 | +0.63 | 3/6 | no clear difference | 1.00 |  |
| Q4 (report) f-LIF vs memoryless LIF, query | nsmt4q_flif | nsmt4q_lif_nomem | 0 / 1 / 1 | +0.58 | 2/6 | no clear difference | 1.02 |  |
| Q4 (report) pop LIF vs memoryless LIF, query | nsmt4q_pop_lif | nsmt4q_lif_nomem | 0 / 2 / 0 | -0.10 | 4/6 | no clear difference | 1.06 |  |
| Q4 (report) pop f-LIF vs memoryless LIF, query | nsmt4q_pop_flif | nsmt4q_lif_nomem | 0 / 2 / 0 | -0.47 | 4/6 | no clear difference | 1.16 |  |
| Q5 (report) f-LIF vs LIF, query | nsmt4q_flif | nsmt4q_lif | 0 / 2 / 0 | -0.05 | 3/6 | no clear difference | 1.01 |  |
| Q5 (report) pop LIF vs LIF, query | nsmt4q_pop_lif | nsmt4q_lif | 0 / 2 / 0 | -0.71 | 4/6 | no clear difference | 1.05 |  |
| Q5 (report) pop f-LIF vs LIF, query | nsmt4q_pop_flif | nsmt4q_lif | 1 / 1 / 0 | -1.08 | 5/6 | better | 1.16 |  |
