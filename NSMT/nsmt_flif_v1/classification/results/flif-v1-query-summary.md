# Query batch (Neocortex as the attention query) — summary

See the module docstring for names, sources and rules. Exploratory (one CinC fold, 3 seeds).

## CinC 2017 fold 0, 90 epochs (test score %, best-validation weights)

| question | A | B | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|---|---|
| Q1 query vs replace, LIF (main) | nsmt4q_lif | nsmt4_lif | | | | | incomplete | | |
| Q1 query vs replace, f-LIF (main) | nsmt4q_flif | nsmt4_flif | | | | | incomplete | | |
| Q1 query vs replace, pop LIF (main) | nsmt4q_pop_lif | nsmt4_pop_lif | | | | | incomplete | | |
| Q1 query vs replace, pop f-LIF (main) | nsmt4q_pop_flif | nsmt4_pop_flif | | | | | incomplete | | |
| Q1 (report) query vs replace, memoryless LIF | nsmt4q_lif_nomem | nsmt4_lif_nomem | | | | | incomplete | | |
| Q2 query vs append (B form), LIF | nsmt4q_lif | spkneo_lif | | | | | incomplete | | |
| Q2 query vs append (B form), f-LIF | nsmt4q_flif | spkneo_flif | | | | | incomplete | | |
| Q2 query vs append (B form), pop LIF | nsmt4q_pop_lif | spkneo_pop_lif | | | | | incomplete | | |
| Q2 query vs append (B form), pop f-LIF | nsmt4q_pop_flif | spkneo_pop_flif | | | | | incomplete | | |
| Q3 query vs Spikformer-style, LIF (criterion 2) | nsmt4q_lif | spk_ssa | | | | | incomplete | | |
| Q3 query vs Spikformer-style, f-LIF (criterion 2) | nsmt4q_flif | spk_ssa | | | | | incomplete | | |
| Q3 query vs Spikformer-style, pop LIF (criterion 2) | nsmt4q_pop_lif | spk_ssa | | | | | incomplete | | |
| Q3 query vs Spikformer-style, pop f-LIF (criterion 2) | nsmt4q_pop_flif | spk_ssa | | | | | incomplete | | |
| Q4 T = N memory vs memoryless, query LIF (main) | nsmt4q_lif | nsmt4q_lif_nomem | | | | | incomplete | | |
| Q4 (report) f-LIF vs memoryless LIF, query | nsmt4q_flif | nsmt4q_lif_nomem | | | | | incomplete | | |
| Q4 (report) pop LIF vs memoryless LIF, query | nsmt4q_pop_lif | nsmt4q_lif_nomem | | | | | incomplete | | |
| Q4 (report) pop f-LIF vs memoryless LIF, query | nsmt4q_pop_flif | nsmt4q_lif_nomem | | | | | incomplete | | |
| Q5 (report) f-LIF vs LIF, query | nsmt4q_flif | nsmt4q_lif | | | | | incomplete | | |
| Q5 (report) pop LIF vs LIF, query | nsmt4q_pop_lif | nsmt4q_lif | | | | | incomplete | | |
| Q5 (report) pop f-LIF vs LIF, query | nsmt4q_pop_flif | nsmt4q_lif | | | | | incomplete | | |

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
