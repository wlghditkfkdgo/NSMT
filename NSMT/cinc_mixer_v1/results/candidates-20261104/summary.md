# Candidate comparison (report only, 2026-11-04)

Sources: see the module docstring. Values are recomputed from RESULT lines.

## CinC fold 0 (test CinC score %, best-validation weights)

| A vs B | A mean | B mean | A − B %p (seed 42, 7, 2026) | verdict | energy A/B |
|---|---:|---:|---|---|---:|
| muxlin (30 ep) vs Spikformer-style tuned (lr 2e-3, 30 ep) | 68.52 | 66.36 | +2.16 (+2.86, +2.54, +1.09) | better | 0.98 |
| muxlin (30 ep) vs Spikformer-style default (lr 1e-3, 30 ep) | 68.52 | 65.60 | +2.92 (+2.24, +3.10, +3.43) | better | 1.02 |
| nsmt4q_pop_lif (90 ep) vs Spikformer-style tuned | 66.60 | 66.36 | +0.24 (-0.77, +1.53, -0.03) | no clear difference | 1.21 |
| nsmt4q_pop_lif (90 ep) vs Spikformer-style default | 66.60 | 65.60 | +1.00 (-1.39, +2.09, +2.32) | better | 1.25 |
| muxlin vs nsmt4q_pop_lif | 68.52 | 66.60 | +1.92 (+3.63, +1.01, +1.12) | better | 0.81 |
| nsmt4q_pop_flif (90 ep) vs Spikformer-style tuned | 66.47 | 66.36 | +0.11 (+0.26, +1.28, -1.20) | no clear difference | 1.37 |
| nsmt4q_pop_flif (90 ep) vs Spikformer-style default | 66.47 | 65.60 | +0.87 (-0.35, +1.83, +1.14) | better | 1.41 |
| muxlin vs nsmt4q_pop_flif | 68.52 | 66.47 | +2.05 (+2.60, +1.27, +2.29) | better | 0.72 |

## CinC folds 0-4: B vs Spikformer-style default (15 pairs; reproduces the other session's order summary)

| model | 5-fold mean | Spikformer-style mean | pooled %p | 95 % t interval | pairs ahead | energy A/B |
|---|---:|---:|---:|---|---:|---:|
| nsmt4q_pop_lif | 67.73 | 65.74 | +1.99 | [+0.97, +3.00] | 12/15 | 1.25 |
| nsmt4q_pop_flif | 67.58 | 65.74 | +1.84 | [+0.93, +2.75] | 13/15 | 1.42 |

## ETT (ETTh2 / ETTm2 x H96 / H720, test MSE; R = mean of per-cell seed-mean r, r = 100 (MSE_A − MSE_B) / MSE_B)

| A vs B | R % | lower pairs | verdict | energy A/B | per cell r % (seed 42, 7, 2026) |
|---|---:|---:|---|---:|---|
| muxlin tuned vs Spikformer-style default | +0.40 | 4/12 | no clear difference | 0.98 | ETTh2 H96 +1.67 (+2.37, +0.39, +2.25) w; ETTh2 H720 -0.12 (+0.83, -0.55, -0.64) .; ETTm2 H96 +0.04 (-0.56, +0.16, +0.52) .; ETTm2 H720 +0.00 (+0.39, -0.42, +0.03) . |
| muxlin tuned vs Spikformer-style tuned | +0.45 | 3/12 | no clear difference | 0.98 | ETTh2 H96 +1.67 (+2.37, +0.39, +2.25) w; ETTh2 H720 -0.12 (+0.83, -0.55, -0.64) .; ETTm2 H96 +0.47 (+0.19, +0.43, +0.77) .; ETTm2 H720 -0.23 (-0.89, +0.07, +0.12) . |
| nsmt4q_pop_lif vs Spikformer-style default | -0.68 | 9/12 | no clear difference | 1.24 | ETTh2 H96 +0.30 (-1.68, +1.84, +0.76) .; ETTh2 H720 -2.07 (-1.76, -1.44, -3.03) b; ETTm2 H96 -0.80 (-1.25, -0.54, -0.62) .; ETTm2 H720 -0.16 (-0.33, -0.49, +0.33) . |
| nsmt4q_pop_lif vs Spikformer-style tuned | -0.63 | 8/12 | no clear difference | 1.23 | ETTh2 H96 +0.30 (-1.68, +1.84, +0.76) .; ETTh2 H720 -2.07 (-1.76, -1.44, -3.03) b; ETTm2 H96 -0.38 (-0.50, -0.27, -0.37) .; ETTm2 H720 -0.39 (-1.59, +0.01, +0.41) . |
| nsmt4q_pop_flif vs Spikformer-style default | -0.85 | 9/12 | no clear difference | 1.50 | ETTh2 H96 -0.44 (-2.66, +1.16, +0.19) .; ETTh2 H720 -2.08 (-1.54, -1.61, -3.07) b; ETTm2 H96 -0.53 (-1.03, -0.86, +0.30) .; ETTm2 H720 -0.37 (-0.29, -0.81, -0.02) . |
| nsmt4q_pop_flif vs Spikformer-style tuned | -0.81 | 8/12 | no clear difference | 1.50 | ETTh2 H96 -0.44 (-2.66, +1.16, +0.19) .; ETTh2 H720 -2.08 (-1.54, -1.61, -3.07) b; ETTm2 H96 -0.10 (-0.28, -0.59, +0.56) .; ETTm2 H720 -0.60 (-1.55, -0.32, +0.07) . |
| nsmt4q_pop_lif vs muxlin tuned | -1.07 | 10/12 | better | 1.26 | ETTh2 H96 -1.33 (-3.96, +1.44, -1.46) b; ETTh2 H720 -1.95 (-2.56, -0.89, -2.40) b; ETTm2 H96 -0.84 (-0.69, -0.70, -1.13) .; ETTm2 H720 -0.16 (-0.71, -0.07, +0.30) . |

