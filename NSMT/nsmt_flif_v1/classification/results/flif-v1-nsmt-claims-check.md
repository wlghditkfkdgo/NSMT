# What the original NSMT paper can claim, on the runs we have

See the module docstring. r % of test MSE, seeds 42 / 7 / 2026.

## ETT

| A vs B | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B |
|---|---|---|---|---|---|---|---|---|
| NSMT vs Spikformer-style (same rule) | -0.27 (no clear difference) | -1.66 (better) | -1.73 (better) | -0.50 (no clear difference) | -1.04 | 10/12 | better | 6.25 |
| NSMT vs hippo-only T = 24 (same rule) | +1.42 (worse) | +1.00 (worse) | +0.26 (no clear difference) | +0.07 (no clear difference) | +0.69 | 2/12 | no clear difference | 0.99 |
| hippo-only T = 24 vs Spikformer-style (same rule) | -1.66 (better) | -2.63 (better) | -1.99 (better) | -0.57 (no clear difference) | -1.71 | 11/12 | better | 6.33 |
| NSMT vs hippo-only T = 24 (paper rule) | -0.09 (no clear difference) | -1.05 (better) | -0.09 (no clear difference) | +0.02 (no clear difference) | -0.30 | 7/12 | no clear difference | 0.97 |

## NSMT vs the linear baseline (r % of test MSE, mean of 3 seeds)

| rule | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean |
|---|---|---|---|---|---|
| same rule | -0.67 | -1.31 | -4.63 | -1.57 | -2.04 |
| paper rule | -2.77 | -2.63 | -4.28 | -1.87 | -2.89 |

## MIT-BIH (macro-F1 %p, NSMT reproduction − B, 42 / 7 / 2026)

| B | per seed | mean | verdict | energy NSMT / B |
|---|---|---|---|---|
| Spikformer-style | -0.78 / -0.68 / -0.47 | -0.64 | worse | 2.50 |
| hippo-only | -0.42 / -0.25 / -0.07 | -0.25 | no clear difference | 0.92 |
| no_neo (no neocortex, no cross-attention) | +0.86 / +1.29 / +1.40 | +1.18 | better | 1.83 |

## Table 7 of the paper, recomputed from its own energy columns

| dataset | w/o align uJ | w/ align uJ | w/ / w/o | real reduction % | (w/o − w/) / w/ % | printed "Improvement" % |
|---|---|---|---|---|---|---|
| ETTh2 | 53.32 | 37.36 | 0.701 | 29.9 | 42.72 | 42.74 |
| ETTm1 | 67.65 | 44.28 | 0.655 | 34.5 | 52.78 | 52.79 |
| ETTm2 | 62.49 | 41.36 | 0.662 | 33.8 | 51.09 | 51.10 |
