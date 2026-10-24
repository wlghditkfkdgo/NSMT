# CinC folds 0-4 `flif-v1-rhythm-folds-cinc` — summary

Best-validation weights on each fold's test set; CinC score (%); fold 0 from the earlier suites.

| condition | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | all (15) | macro-F1 all | total µJ |
|---|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 65.60 | 66.30 | 65.89 | 64.44 | 66.46 | 65.74 ± 1.10 | 61.39 | 126.8 |
| linear fusion of spikes, LIF | 61.37 | 63.61 | 62.26 | 61.50 | 62.28 | 62.20 ± 1.21 | 57.41 | 31.6 |
| charges + |H_f - H_r|, LIF | 65.61 | 67.32 | 64.54 | 64.61 | 65.59 | 65.53 ± 1.57 | 60.03 | 267.4 |

## Comparisons (%p)

| question | A | B | per fold (mean of 3 seeds) and verdict | pooled mean [95% CI] | A > B pairs |
|---|---|---|---|---|---|
| F1 charges + difference vs Spikformer | charges + |H_f - H_r|, LIF | Spikformer-style SSA (T=4) | f0 +0.02 no clear difference; f1 +1.02 better; f2 -1.36 worse; f3 +0.17 no clear difference; f4 -0.87 worse | -0.21 [-1.11, +0.70] | 6/15 |
| F2 charges + difference vs spike linear fusion | charges + |H_f - H_r|, LIF | linear fusion of spikes, LIF | f0 +4.24 better; f1 +3.71 better; f2 +2.28 better; f3 +3.11 better; f4 +3.32 better | +3.33 [+2.63, +4.03] | 15/15 |
| F3 spike linear fusion vs Spikformer | linear fusion of spikes, LIF | Spikformer-style SSA (T=4) | f0 -4.22 worse; f1 -2.70 worse; f2 -3.63 worse; f3 -2.94 worse; f4 -4.19 worse | -3.54 [-4.29, -2.78] | 0/15 |
