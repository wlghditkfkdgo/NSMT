# Dual-path batch ([NSMT hippocampus T=4] and B form) — summary

See the module docstring for names, sources and rules. Exploratory (one CinC fold, 3 seeds).

## CinC 2017 fold 0 (test score %, best-validation weights; dual-path models and Spikformer-style at 90 epochs)

| question | A | B | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|---|---|
| R1 replace vs append, LIF (main) | nsmt4_lif | spkneo_lif | 62.97 | 65.49 | -3.12 / -2.39 / -2.07 | -2.52 | worse | 0.84 |  |
| R1 replace vs append, f-LIF (main) | nsmt4_flif | spkneo_flif | 60.26 | 64.25 | -4.83 / -4.33 / -2.82 | -3.99 | worse | 0.82 |  |
| R1 replace vs append, pop LIF (main) | nsmt4_pop_lif | spkneo_pop_lif | 63.51 | 65.38 | -4.13 / -1.84 / +0.37 | -1.87 | worse | 0.83 |  |
| R1 replace vs append, pop f-LIF (main) | nsmt4_pop_flif | spkneo_pop_flif | 60.63 | 65.28 | -3.57 / -4.33 / -6.05 | -4.65 | worse | 0.85 |  |
| R1 (report) replace vs append, memoryless LIF | nsmt4_lif_nomem | spkneo_lif_nomem | 59.12 | 66.04 | -6.95 / -6.35 / -7.45 | -6.92 | worse | 0.83 |  |
| R2 [NSMT T=4] vs Spikformer-style, LIF (criterion 2) | nsmt4_lif | spk_ssa | 62.97 | 64.89 | -2.88 / -1.79 / -1.11 | -1.93 | worse | 1.10 | not met |
| R2 [NSMT T=4] vs Spikformer-style, f-LIF (criterion 2) | nsmt4_flif | spk_ssa | 60.26 | 64.89 | -4.82 / -6.60 / -2.50 | -4.64 | worse | 1.10 | not met |
| R2 [NSMT T=4] vs Spikformer-style, pop LIF (criterion 2) | nsmt4_pop_lif | spk_ssa | 63.51 | 64.89 | -3.77 / -0.66 / +0.26 | -1.39 | worse | 1.13 | not met |
| R2 [NSMT T=4] vs Spikformer-style, pop f-LIF (criterion 2) | nsmt4_pop_flif | spk_ssa | 60.63 | 64.89 | -5.40 / -4.36 / -3.03 | -4.26 | worse | 1.28 | not met |
| R2 B form vs Spikformer-style, LIF (criterion 2) [= earlier D1] | spkneo_lif | spk_ssa | 65.49 | 64.89 | +0.24 / +0.60 / +0.96 | +0.60 | better | 1.31 | met |
| R2 B form vs Spikformer-style, f-LIF (criterion 2) [= earlier D1] | spkneo_flif | spk_ssa | 64.25 | 64.89 | +0.01 / -2.27 / +0.32 | -0.64 | no clear difference | 1.33 | not met |
| R2 B form vs Spikformer-style, pop LIF (criterion 2) [= earlier D1] | spkneo_pop_lif | spk_ssa | 65.38 | 64.89 | +0.36 / +1.19 / -0.11 | +0.48 | no clear difference | 1.37 | not met |
| R2 B form vs Spikformer-style, pop f-LIF (criterion 2) [= earlier D1] | spkneo_pop_flif | spk_ssa | 65.28 | 64.89 | -1.83 / -0.03 / +3.02 | +0.39 | no clear difference | 1.50 | not met |
| R3 T = N memory vs memoryless, [NSMT T=4] LIF | nsmt4_lif | nsmt4_lif_nomem | 62.97 | 59.12 | +3.99 / +3.29 / +4.28 | +3.85 | better | 1.03 |  |
| R3 T = N memory vs memoryless, B form LIF [= earlier D2] | spkneo_lif | spkneo_lif_nomem | 65.49 | 66.04 | +0.15 / -0.68 / -1.10 | -0.54 | worse | 1.02 |  |
| R3 (report) f-LIF vs memoryless LIF, [NSMT T=4] | nsmt4_flif | nsmt4_lif_nomem | 60.26 | 59.12 | +2.05 / -1.52 / +2.89 | +1.14 | better | 1.03 |  |
| R3 (report) pop LIF vs memoryless LIF, [NSMT T=4] | nsmt4_pop_lif | nsmt4_lif_nomem | 63.51 | 59.12 | +3.10 / +4.42 / +5.65 | +4.39 | better | 1.06 |  |
| R3 (report) pop f-LIF vs memoryless LIF, [NSMT T=4] | nsmt4_pop_flif | nsmt4_lif_nomem | 60.63 | 59.12 | +1.47 / +0.72 / +2.36 | +1.51 | better | 1.20 |  |
| R3 (report) f-LIF vs memoryless LIF, B form | spkneo_flif | spkneo_lif_nomem | 64.25 | 66.04 | -0.07 / -3.55 / -1.73 | -1.79 | worse | 1.04 |  |
| R3 (report) pop LIF vs memoryless LIF, B form | spkneo_pop_lif | spkneo_lif_nomem | 65.38 | 66.04 | +0.28 / -0.09 / -2.17 | -0.66 | worse | 1.07 |  |
| R3 (report) pop f-LIF vs memoryless LIF, B form | spkneo_pop_flif | spkneo_lif_nomem | 65.28 | 66.04 | -1.91 / -1.31 / +0.96 | -0.75 | worse | 1.17 |  |
| R4 (report) f-LIF vs LIF, [NSMT T=4] | nsmt4_flif | nsmt4_lif | 60.26 | 62.97 | -1.94 / -4.81 / -1.39 | -2.71 | worse | 1.00 |  |
| R4 (report) pop LIF vs LIF, [NSMT T=4] | nsmt4_pop_lif | nsmt4_lif | 63.51 | 62.97 | -0.89 / +1.13 / +1.37 | +0.54 | better | 1.03 |  |
| R4 (report) pop f-LIF vs LIF, [NSMT T=4] | nsmt4_pop_flif | nsmt4_lif | 60.63 | 62.97 | -2.52 / -2.57 / -1.92 | -2.34 | worse | 1.17 |  |
| R4 (report) f-LIF vs LIF, B form | spkneo_flif | spkneo_lif | 64.25 | 65.49 | -0.23 / -2.87 / -0.63 | -1.24 | worse | 1.02 |  |
| R4 (report) pop LIF vs LIF, B form | spkneo_pop_lif | spkneo_lif | 65.38 | 65.49 | +0.13 / +0.59 / -1.06 | -0.12 | no clear difference | 1.05 |  |
| R4 (report) pop f-LIF vs LIF, B form | spkneo_pop_flif | spkneo_lif | 65.28 | 65.49 | -2.06 / -0.63 / +2.06 | -0.21 | no clear difference | 1.15 |  |
| R6 (report) [NSMT T=4] @90 new settings vs @30 old settings, LIF | nsmt4_lif | nsmt_lif_t4 | 62.97 | 63.42 | -0.16 / -1.21 / +0.01 | -0.45 | no clear difference | 0.97 |  |
| R6 (report) [NSMT T=4] @90 new settings vs @30 old settings, pop f-LIF | nsmt4_pop_flif | nsmt_pop_t4 | 60.63 | 58.66 | +3.17 / +2.95 / -0.22 | +1.97 | better | 1.05 |  |

## MIT-BIH (test macro-F1 %, last-epoch weights)

| question | A | B | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|---|---|
| R1 replace vs append, LIF (main) | nsmt4_lif | spkneo_lif | 91.97 | 91.98 | -0.14 / +0.57 / -0.44 | -0.00 | no clear difference | 0.78 |  |
| R1 replace vs append, f-LIF (main) | nsmt4_flif | spkneo_flif | 91.57 | 91.95 | -0.22 / -0.61 / -0.33 | -0.39 | no clear difference | 0.79 |  |
| R1 replace vs append, pop LIF (main) | nsmt4_pop_lif | spkneo_pop_lif | 91.56 | 92.20 | -0.89 / -0.35 / -0.67 | -0.64 | worse | 0.81 |  |
| R1 replace vs append, pop f-LIF (main) | nsmt4_pop_flif | spkneo_pop_flif | 91.47 | 92.06 | -0.37 / -0.35 / -1.02 | -0.58 | worse | 0.82 |  |
| R1 (report) replace vs append, memoryless LIF | nsmt4_lif_nomem | spkneo_lif_nomem | 92.04 | 92.15 | +0.25 / -0.06 / -0.53 | -0.12 | no clear difference | 0.77 |  |
| R2 [NSMT T=4] vs Spikformer-style, LIF (criterion 2) | nsmt4_lif | spk_ssa | 91.97 | 92.22 | -0.54 / +0.11 / -0.32 | -0.25 | no clear difference | 1.02 | not met |
| R2 [NSMT T=4] vs Spikformer-style, f-LIF (criterion 2) | nsmt4_flif | spk_ssa | 91.57 | 92.22 | -0.69 / -0.77 / -0.51 | -0.66 | worse | 1.01 | not met |
| R2 [NSMT T=4] vs Spikformer-style, pop LIF (criterion 2) | nsmt4_pop_lif | spk_ssa | 91.56 | 92.22 | -0.95 / -0.62 / -0.41 | -0.66 | worse | 1.05 | not met |
| R2 [NSMT T=4] vs Spikformer-style, pop f-LIF (criterion 2) | nsmt4_pop_flif | spk_ssa | 91.47 | 92.22 | -0.87 / -0.42 / -0.95 | -0.75 | worse | 1.15 | not met |
| R2 B form vs Spikformer-style, LIF (criterion 2) | spkneo_lif | spk_ssa | 91.98 | 92.22 | -0.40 / -0.46 / +0.12 | -0.25 | no clear difference | 1.31 | not met |
| R2 B form vs Spikformer-style, f-LIF (criterion 2) | spkneo_flif | spk_ssa | 91.95 | 92.22 | -0.47 / -0.17 / -0.18 | -0.27 | no clear difference | 1.28 | not met |
| R2 B form vs Spikformer-style, pop LIF (criterion 2) | spkneo_pop_lif | spk_ssa | 92.20 | 92.22 | -0.05 / -0.26 / +0.26 | -0.02 | no clear difference | 1.30 | not met |
| R2 B form vs Spikformer-style, pop f-LIF (criterion 2) | spkneo_pop_flif | spk_ssa | 92.06 | 92.22 | -0.50 / -0.07 / +0.07 | -0.16 | no clear difference | 1.41 | not met |
| R3 T = N memory vs memoryless, [NSMT T=4] LIF | nsmt4_lif | nsmt4_lif_nomem | 91.97 | 92.04 | -0.10 / +0.01 / -0.11 | -0.07 | no clear difference | 1.06 |  |
| R3 T = N memory vs memoryless, B form LIF | spkneo_lif | spkneo_lif_nomem | 91.98 | 92.15 | +0.28 / -0.62 / -0.19 | -0.18 | no clear difference | 1.05 |  |
| R3 (report) f-LIF vs memoryless LIF, [NSMT T=4] | nsmt4_flif | nsmt4_lif_nomem | 91.57 | 92.04 | -0.25 / -0.88 / -0.29 | -0.47 | no clear difference | 1.05 |  |
| R3 (report) pop LIF vs memoryless LIF, [NSMT T=4] | nsmt4_pop_lif | nsmt4_lif_nomem | 91.56 | 92.04 | -0.51 / -0.72 / -0.19 | -0.47 | no clear difference | 1.09 |  |
| R3 (report) pop f-LIF vs memoryless LIF, [NSMT T=4] | nsmt4_pop_flif | nsmt4_lif_nomem | 91.47 | 92.04 | -0.43 / -0.52 / -0.74 | -0.56 | worse | 1.20 |  |
| R3 (report) f-LIF vs memoryless LIF, B form | spkneo_flif | spkneo_lif_nomem | 91.95 | 92.15 | +0.22 / -0.33 / -0.49 | -0.20 | no clear difference | 1.03 |  |
| R3 (report) pop LIF vs memoryless LIF, B form | spkneo_pop_lif | spkneo_lif_nomem | 92.20 | 92.15 | +0.63 / -0.43 / -0.06 | +0.05 | no clear difference | 1.04 |  |
| R3 (report) pop f-LIF vs memoryless LIF, B form | spkneo_pop_flif | spkneo_lif_nomem | 92.06 | 92.15 | +0.19 / -0.23 / -0.25 | -0.10 | no clear difference | 1.14 |  |
| R4 (report) f-LIF vs LIF, [NSMT T=4] | nsmt4_flif | nsmt4_lif | 91.57 | 91.97 | -0.15 / -0.89 / -0.19 | -0.41 | no clear difference | 0.99 |  |
| R4 (report) pop LIF vs LIF, [NSMT T=4] | nsmt4_pop_lif | nsmt4_lif | 91.56 | 91.97 | -0.40 / -0.73 / -0.09 | -0.41 | no clear difference | 1.03 |  |
| R4 (report) pop f-LIF vs LIF, [NSMT T=4] | nsmt4_pop_flif | nsmt4_lif | 91.47 | 91.97 | -0.33 / -0.53 / -0.63 | -0.50 | no clear difference | 1.13 |  |
| R4 (report) f-LIF vs LIF, B form | spkneo_flif | spkneo_lif | 91.95 | 91.98 | -0.06 / +0.29 / -0.30 | -0.02 | no clear difference | 0.98 |  |
| R4 (report) pop LIF vs LIF, B form | spkneo_pop_lif | spkneo_lif | 92.20 | 91.98 | +0.35 / +0.20 / +0.14 | +0.23 | no clear difference | 0.99 |  |
| R4 (report) pop f-LIF vs LIF, B form | spkneo_pop_flif | spkneo_lif | 92.06 | 91.98 | -0.09 / +0.39 / -0.05 | +0.08 | no clear difference | 1.08 |  |
| R5 (report) [NSMT T=4] vs Neo-mixer every layer, LIF | nsmt4_lif | mixall_lif_bidir | 91.97 | 90.70 | +1.17 / +1.71 / +0.94 | +1.27 | better | 4.41 |  |
| R5 (report) [NSMT T=4] vs Neo-mixer every layer, f-LIF | nsmt4_flif | mixall_flif_bidir | 91.57 | 91.12 | +0.52 / +0.48 / +0.33 | +0.44 | no clear difference | 4.07 |  |
| R5 (report) [NSMT T=4] vs Neo-mixer every layer, pop LIF | nsmt4_pop_lif | mixall_pop_lif_bidir | 91.56 | 88.84 | +2.53 / +2.72 / +2.93 | +2.73 | better | 2.70 |  |
| R5 (report) [NSMT T=4] vs Neo-mixer every layer, pop f-LIF | nsmt4_pop_flif | mixall_pop_flif_bidir | 91.47 | 89.70 | +1.60 / +2.57 / +1.16 | +1.78 | better | 1.81 |  |
| R5 (report) B form vs Neo-mixer every layer, LIF | spkneo_lif | mixall_lif_bidir | 91.98 | 90.70 | +1.31 / +1.13 / +1.39 | +1.28 | better | 5.65 |  |
| R5 (report) B form vs Neo-mixer every layer, f-LIF | spkneo_flif | mixall_flif_bidir | 91.95 | 91.12 | +0.74 / +1.09 / +0.66 | +0.83 | better | 5.17 |  |
| R5 (report) B form vs Neo-mixer every layer, pop LIF | spkneo_pop_lif | mixall_pop_lif_bidir | 92.20 | 88.84 | +3.43 / +3.07 / +3.60 | +3.37 | better | 3.34 |  |
| R5 (report) B form vs Neo-mixer every layer, pop f-LIF | spkneo_pop_flif | mixall_pop_flif_bidir | 92.06 | 89.70 | +1.97 / +2.92 / +2.18 | +2.36 | better | 2.21 |  |

## ETT forecasting, ETTh2 + ETTm2 x H96 / H720 (test MSE; r = 100 (A − B) / B, %) — T_h-averaged head energy recounted

| question | A | B | cells b / . / w | R | pairs A < B | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|---|
| R1 replace vs append, LIF (main) | nsmt4_lif | spkneo_lif | 0 / 3 / 1 | -0.00 | 7/12 | no clear difference | 0.83 |  |
| R1 replace vs append, f-LIF (main) | nsmt4_flif | spkneo_flif | 0 / 3 / 1 | +0.15 | 7/12 | no clear difference | 0.83 |  |
| R1 replace vs append, pop LIF (main) | nsmt4_pop_lif | spkneo_pop_lif | 2 / 2 / 0 | -0.99 | 9/12 | no clear difference | 0.84 |  |
| R1 replace vs append, pop f-LIF (main) | nsmt4_pop_flif | spkneo_pop_flif | 0 / 4 / 0 | -0.17 | 6/12 | no clear difference | 0.87 |  |
| R1 (report) replace vs append, memoryless LIF | nsmt4_lif_nomem | spkneo_lif_nomem | 0 / 4 / 0 | -0.52 | 10/12 | no clear difference | 0.83 |  |
| R2 [NSMT T=4] vs Spikformer-style, LIF (criterion 2) | nsmt4_lif | spk_ssa | 0 / 4 / 0 | -0.09 | 7/12 | no clear difference | 1.08 | not met |
| R2 [NSMT T=4] vs Spikformer-style, f-LIF (criterion 2) | nsmt4_flif | spk_ssa | 1 / 3 / 0 | -0.23 | 7/12 | no clear difference | 1.11 | not met |
| R2 [NSMT T=4] vs Spikformer-style, pop LIF (criterion 2) | nsmt4_pop_lif | spk_ssa | 1 / 3 / 0 | -1.06 | 11/12 | better | 1.22 | met |
| R2 [NSMT T=4] vs Spikformer-style, pop f-LIF (criterion 2) | nsmt4_pop_flif | spk_ssa | 1 / 3 / 0 | -0.34 | 9/12 | no clear difference | 1.48 | not met |
| R2 B form vs Spikformer-style, LIF (criterion 2) | spkneo_lif | spk_ssa | 1 / 3 / 0 | -0.08 | 5/12 | no clear difference | 1.30 | not met |
| R2 B form vs Spikformer-style, f-LIF (criterion 2) | spkneo_flif | spk_ssa | 2 / 2 / 0 | -0.37 | 6/12 | no clear difference | 1.34 | not met |
| R2 B form vs Spikformer-style, pop LIF (criterion 2) | spkneo_pop_lif | spk_ssa | 0 / 3 / 1 | -0.06 | 8/12 | no clear difference | 1.45 | not met |
| R2 B form vs Spikformer-style, pop f-LIF (criterion 2) | spkneo_pop_flif | spk_ssa | 0 / 4 / 0 | -0.18 | 8/12 | no clear difference | 1.71 | not met |
| R3 T = N memory vs memoryless, [NSMT T=4] LIF | nsmt4_lif | nsmt4_lif_nomem | 0 / 3 / 1 | +0.56 | 6/12 | no clear difference | 1.01 |  |
| R3 T = N memory vs memoryless, B form LIF | spkneo_lif | spkneo_lif_nomem | 0 / 4 / 0 | +0.04 | 7/12 | no clear difference | 1.01 |  |
| R3 (report) f-LIF vs memoryless LIF, [NSMT T=4] | nsmt4_flif | nsmt4_lif_nomem | 0 / 3 / 1 | +0.42 | 3/12 | no clear difference | 1.04 |  |
| R3 (report) pop LIF vs memoryless LIF, [NSMT T=4] | nsmt4_pop_lif | nsmt4_lif_nomem | 1 / 2 / 1 | -0.42 | 8/12 | no clear difference | 1.14 |  |
| R3 (report) pop f-LIF vs memoryless LIF, [NSMT T=4] | nsmt4_pop_flif | nsmt4_lif_nomem | 0 / 3 / 1 | +0.31 | 4/12 | no clear difference | 1.39 |  |
| R3 (report) f-LIF vs memoryless LIF, B form | spkneo_flif | spkneo_lif_nomem | 1 / 3 / 0 | -0.25 | 6/12 | no clear difference | 1.04 |  |
| R3 (report) pop LIF vs memoryless LIF, B form | spkneo_pop_lif | spkneo_lif_nomem | 0 / 3 / 1 | +0.07 | 8/12 | no clear difference | 1.12 |  |
| R3 (report) pop f-LIF vs memoryless LIF, B form | spkneo_pop_flif | spkneo_lif_nomem | 0 / 4 / 0 | -0.05 | 5/12 | no clear difference | 1.32 |  |
| R4 (report) f-LIF vs LIF, [NSMT T=4] | nsmt4_flif | nsmt4_lif | 0 / 4 / 0 | -0.13 | 3/12 | no clear difference | 1.03 |  |
| R4 (report) pop LIF vs LIF, [NSMT T=4] | nsmt4_pop_lif | nsmt4_lif | 1 / 3 / 0 | -0.96 | 10/12 | no clear difference | 1.13 |  |
| R4 (report) pop f-LIF vs LIF, [NSMT T=4] | nsmt4_pop_flif | nsmt4_lif | 0 / 4 / 0 | -0.24 | 7/12 | no clear difference | 1.38 |  |
| R4 (report) f-LIF vs LIF, B form | spkneo_flif | spkneo_lif | 0 / 4 / 0 | -0.28 | 4/12 | no clear difference | 1.03 |  |
| R4 (report) pop LIF vs LIF, B form | spkneo_pop_lif | spkneo_lif | 1 / 2 / 1 | +0.04 | 7/12 | no clear difference | 1.11 |  |
| R4 (report) pop f-LIF vs LIF, B form | spkneo_pop_flif | spkneo_lif | 0 / 4 / 0 | -0.08 | 7/12 | no clear difference | 1.31 |  |
| R5 (report) [NSMT T=4] vs Neo-mixer every layer, LIF | nsmt4_lif | mixall_lif_bidir | 2 / 2 / 0 | -1.10 | 11/12 | better | 5.33 |  |
| R5 (report) [NSMT T=4] vs Neo-mixer every layer, f-LIF | nsmt4_flif | mixall_flif_bidir | 1 / 3 / 0 | -0.67 | 8/12 | no clear difference | 3.87 |  |
| R5 (report) [NSMT T=4] vs Neo-mixer every layer, pop LIF | nsmt4_pop_lif | mixall_pop_lif_bidir | 2 / 2 / 0 | -2.92 | 12/12 | better | 2.26 |  |
| R5 (report) [NSMT T=4] vs Neo-mixer every layer, pop f-LIF | nsmt4_pop_flif | mixall_pop_flif_bidir | 2 / 2 / 0 | -0.73 | 8/12 | no clear difference | 1.30 |  |
| R5 (report) B form vs Neo-mixer every layer, LIF | spkneo_lif | mixall_lif_bidir | 2 / 2 / 0 | -1.08 | 7/12 | no clear difference | 6.46 |  |
| R5 (report) B form vs Neo-mixer every layer, f-LIF | spkneo_flif | mixall_flif_bidir | 2 / 2 / 0 | -0.80 | 7/12 | no clear difference | 4.66 |  |
| R5 (report) B form vs Neo-mixer every layer, pop LIF | spkneo_pop_lif | mixall_pop_lif_bidir | 2 / 2 / 0 | -1.95 | 11/12 | better | 2.69 |  |
| R5 (report) B form vs Neo-mixer every layer, pop f-LIF | spkneo_pop_flif | mixall_pop_flif_bidir | 2 / 2 / 0 | -0.56 | 7/12 | no clear difference | 1.49 |  |

### Per-cell mean r (%)

| A vs B | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| nsmt4_lif − spkneo_lif | +0.3 . | +1.0 w | -0.6 . | -0.7 . |
| nsmt4_flif − spkneo_flif | +1.4 w | -0.1 . | -0.1 . | -0.6 . |
| nsmt4_pop_lif − spkneo_pop_lif | -1.0 b | -2.2 b | -0.3 . | -0.5 . |
| nsmt4_pop_flif − spkneo_pop_flif | +0.6 . | -0.5 . | +0.0 . | -0.8 . |
| nsmt4_lif_nomem − spkneo_lif_nomem | -0.8 . | -0.1 . | -0.7 . | -0.5 . |
| nsmt4_lif − spk_ssa | +0.3 . | -0.3 . | +0.1 . | -0.4 . |
| nsmt4_flif − spk_ssa | +0.2 . | -1.3 b | -0.0 . | +0.2 . |
| nsmt4_pop_lif − spk_ssa | +0.2 . | -2.9 b | -0.8 . | -0.8 . |
| nsmt4_pop_flif − spk_ssa | +0.4 . | -1.3 b | -0.2 . | -0.3 . |
| spkneo_lif − spk_ssa | -0.0 . | -1.3 b | +0.7 . | +0.3 . |
| spkneo_flif − spk_ssa | -1.2 b | -1.1 b | +0.1 . | +0.7 . |
| spkneo_pop_lif − spk_ssa | +1.3 w | -0.7 . | -0.6 . | -0.2 . |
| spkneo_pop_flif − spk_ssa | -0.2 . | -0.7 . | -0.3 . | +0.5 . |
| nsmt4_lif − nsmt4_lif_nomem | +1.1 w | +0.7 . | +0.9 . | -0.4 . |
| spkneo_lif − spkneo_lif_nomem | -0.0 . | -0.4 . | +0.8 . | -0.2 . |
| nsmt4_flif − nsmt4_lif_nomem | +1.0 w | -0.3 . | +0.8 . | +0.2 . |
| nsmt4_pop_lif − nsmt4_lif_nomem | +1.1 w | -1.9 b | -0.0 . | -0.8 . |
| nsmt4_pop_flif − nsmt4_lif_nomem | +1.2 w | -0.2 . | +0.6 . | -0.3 . |
| spkneo_flif − spkneo_lif_nomem | -1.2 b | -0.2 . | +0.2 . | +0.2 . |
| spkneo_pop_lif − spkneo_lif_nomem | +1.3 w | +0.2 . | -0.5 . | -0.7 . |
| spkneo_pop_flif − spkneo_lif_nomem | -0.2 . | +0.2 . | -0.2 . | +0.0 . |
| nsmt4_flif − nsmt4_lif | -0.0 . | -1.0 . | -0.1 . | +0.6 . |
| nsmt4_pop_lif − nsmt4_lif | -0.0 . | -2.6 b | -0.9 . | -0.4 . |
| nsmt4_pop_flif − nsmt4_lif | +0.1 . | -0.9 . | -0.3 . | +0.1 . |
| spkneo_flif − spkneo_lif | -1.2 . | +0.2 . | -0.6 . | +0.5 . |
| spkneo_pop_lif − spkneo_lif | +1.3 w | +0.6 . | -1.3 b | -0.5 . |
| spkneo_pop_flif − spkneo_lif | -0.2 . | +0.6 . | -1.0 . | +0.2 . |
| nsmt4_lif − mixall_lif_bidir | -1.7 b | -1.9 b | -0.5 . | -0.3 . |
| nsmt4_flif − mixall_flif_bidir | -0.6 . | -1.6 b | -0.5 . | -0.0 . |
| nsmt4_pop_lif − mixall_pop_lif_bidir | -4.5 b | -5.5 b | -0.7 . | -1.0 . |
| nsmt4_pop_flif − mixall_pop_flif_bidir | -1.1 b | -1.6 b | -0.1 . | -0.1 . |
| spkneo_lif − mixall_lif_bidir | -2.0 b | -2.9 b | +0.2 . | +0.4 . |
| spkneo_flif − mixall_flif_bidir | -2.0 b | -1.4 b | -0.4 . | +0.6 . |
| spkneo_pop_lif − mixall_pop_lif_bidir | -3.5 b | -3.4 b | -0.4 . | -0.4 . |
| spkneo_pop_flif − mixall_pop_flif_bidir | -1.8 b | -1.1 b | -0.1 . | +0.7 . |
