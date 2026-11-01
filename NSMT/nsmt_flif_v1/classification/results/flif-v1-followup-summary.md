# CinC follow-up (query folds 1-4, population memoryless controls, Neo-mixer reset) — summary

See the module docstring for sources and rules. Test CinC score (%), best-validation weights.

## A. Query form vs Spikformer-style @30 over folds 0-4 (15 fold x seed pairs)

| model | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 | 5-fold mean | pooled A − B | 95 % t interval | A > B pairs | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Spikformer-style @30 | 65.60 | 66.30 | 65.89 | 64.44 | 66.46 | 65.74 | | | | |
| query pop_lif | 66.60 | 67.81 | 66.84 | 68.30 | 69.09 | 67.73 | +1.99 | [+0.97, +3.00] | 12/15 | better |
| query pop_flif | 66.47 | 68.91 | 65.78 | 67.95 | 68.77 | 67.58 | +1.84 | [+0.93, +2.75] | 13/15 | better |

### per-fold verdicts (query − Spikformer-style @30; >= +0.5 %p and 2 of 3 seeds)

| model | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| query pop_lif | +1.00 better | +1.51 better | +0.95 better | +3.86 better | +2.62 better |
| query pop_flif | +0.87 better | +2.61 better | -0.11 no clear difference | +3.51 better | +2.31 better |

### report only (added after the results): folds 1-4 only (the query form was chosen on fold 0), macro-F1, energy

| model | folds 1-4 A − B (12 pairs) | 95 % t interval (11 df) | A > B pairs | 5-fold macro-F1 A − B | A > B pairs | energy A/B (5 folds) |
|---|---|---|---|---|---|---|
| query pop_lif | +2.23 | [+1.10, +3.36] | 10/12 | +0.98 | 8/15 | 1.25 |
| query pop_flif | +2.08 | [+1.00, +3.16] | 11/12 | +0.88 | 10/15 | 1.42 |

per-class F1, A − B over the 15 pairs (%p, A > B pairs; score = mean of N / AF / Other, macro-F1 adds Noisy)

| model | Normal | AF | Other | Noisy |
|---|---|---|---|---|
| query pop_lif | -0.62 (4/15) | +5.55 (14/15) | +1.03 (11/15) | -2.06 (7/15) |
| query pop_flif | -0.54 (2/15) | +4.71 (14/15) | +1.35 (12/15) | -1.99 (6/15) |

## B. T = N memory for the population neurons in the query form (fold 0)

| comparison | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B |
|---|---|---|---|---|---|---|
| B (main) query population LIF − its memoryless control | 66.60 | 65.22 | -0.59 / +1.81 / +2.93 | +1.38 | better | 1.07 |
| B (main) query population f-LIF − its memoryless control | 66.47 | 65.28 | +0.02 / +2.13 / +1.42 | +1.19 | better | 1.19 |
| B (report) memoryless population LIF − memoryless LIF | 65.22 | 64.94 | +2.21 / -0.77 / -0.60 | +0.28 | no clear difference | 1.03 |
| B (report) memoryless population f-LIF − memoryless LIF | 65.28 | 64.94 | +2.64 / -1.35 / -0.26 | +0.34 | no clear difference | 1.05 |

## C. Neo-mixer with LIF in every layer: spikeDE reset zero vs default (fold 0)

| comparison | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B |
|---|---|---|---|---|---|---|
| C (main) zero − default reset | 61.19 | 61.92 | -0.90 / -1.16 / -0.12 | -0.73 | worse | 0.92 |
| C (report) default reset − Spikformer-style @90 | 61.92 | 64.89 | -2.53 / -3.57 / -2.84 | -2.98 | worse | 0.23 |
| C (report) zero reset − Spikformer-style @90 | 61.19 | 64.89 | -3.43 / -4.73 / -2.95 | -3.71 | worse | 0.22 |
| C (report) default reset − Spikformer-style @30 | 61.92 | 65.60 | -4.14 / -3.01 / -3.90 | -3.68 | worse | 0.24 |
| C (report) zero reset − Spikformer-style @30 | 61.19 | 65.60 | -5.04 / -4.17 / -4.01 | -4.41 | worse | 0.22 |
| C (report) zero reset − mixer-only Neo-mixer LIF @90 | 61.19 | 62.37 | +0.86 / -2.56 / -1.85 | -1.19 | worse | 0.97 |
