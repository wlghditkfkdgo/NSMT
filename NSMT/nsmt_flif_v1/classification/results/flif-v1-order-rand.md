# Test-time token-order shuffle with a random order per row (stored models, no training)

See the docstring of order_eval_rand.py. User request 11-20. Bias on = flif-v1-*-sde, bias off = flif-v1-*-nope (the default since 11-20). R = 5 repeats (generator seed 1000 + r); each row of a batch has its own random token order. Run shas: 11c83b385, 96fb4dc2a.

- body: tokens shuffled for both paths, the hippocampus output put back before the head (does the body use the order?)
- neo: only the Neocortex reads the tokens in the random order (base only)
- all: as body, not put back (ETT: the flat head also reads the tokens at the wrong positions)

## ETT (r = shuffled test MSE / normal test MSE - 1, %; mean of the R repeats per run, then over the 12 runs = 4 cells x 3 seeds)

| model | bias | normal MSE (mean of 12) | body r % (min .. max of 12) | runs r > 0 | neo r % (min .. max) | all r % | largest repeat sd of body r |
|---|---|---|---|---|---|---|---|
| Spikformer(DE) | off | 0.3219 | +0.00 (+0.00 .. +0.00) | 0/12 | — | +21.7 | 0.00 |
| Spikformer(DE) | on | 0.3244 | +0.39 (-0.25 .. +1.54) | 7/12 | — | +21.2 | 0.05 |
| base pop LIF | off | 0.3215 | +0.30 (-1.84 .. +2.10) | 8/12 | +0.30 (-1.84 .. +2.10) | +21.0 | 0.05 |
| base pop LIF | on | 0.3217 | +0.86 (-1.92 .. +6.61) | 7/12 | +0.62 (-1.92 .. +4.95) | +21.0 | 0.05 |
| base pop f-LIF | off | 0.3218 | +0.91 (-0.46 .. +2.95) | 10/12 | +0.91 (-0.46 .. +2.95) | +21.1 | 0.06 |
| base pop f-LIF | on | 0.3220 | +1.13 (-0.37 .. +3.60) | 11/12 | +1.01 (-0.36 .. +3.57) | +21.3 | 0.08 |

### ETT body r % per cell (mean of 3 seeds; seeds in brackets)

| model | bias | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|---|
| Spikformer(DE) | off | +0.00 (+0.00 / +0.00 / +0.00) | +0.00 (+0.00 / +0.00 / +0.00) | +0.00 (+0.00 / +0.00 / +0.00) | +0.00 (+0.00 / +0.00 / +0.00) |
| Spikformer(DE) | on | -0.13 (-0.25 / -0.10 / -0.03) | +0.19 (+0.49 / +0.08 / -0.00) | +1.32 (+1.37 / +1.54 / +1.06) | +0.18 (-0.05 / +0.12 / +0.48) |
| base pop LIF | off | -0.41 (+0.78 / -1.84 / -0.16) | -0.18 (-0.79 / +0.48 / -0.25) | +1.44 (+2.10 / +1.19 / +1.02) | +0.34 (+0.17 / +0.63 / +0.21) |
| base pop LIF | on | -1.34 (-1.12 / -1.92 / -0.98) | +0.06 (-0.76 / +1.27 / -0.33) | +4.16 (+6.61 / +3.34 / +2.52) | +0.58 (+0.20 / +1.10 / +0.45) |
| base pop f-LIF | off | +1.22 (+2.11 / +2.00 / -0.46) | +0.98 (-0.17 / +2.95 / +0.16) | +1.03 (+0.01 / +0.75 / +2.32) | +0.43 (+0.28 / +0.51 / +0.49) |
| base pop f-LIF | on | +1.54 (+1.54 / +1.89 / +1.19) | +1.70 (+0.78 / +3.60 / +0.74) | +0.80 (+1.11 / -0.37 / +1.67) | +0.47 (+0.15 / +0.56 / +0.72) |

## MIT-BIH (macro-F1 %; drop = shuffled - normal, %p, mean of the R repeats per seed; changed = share of test beats whose prediction changes, %)

| model | bias | normal | body drop %p (seed 42 / 7 / 2026) | mean | changed % | neo drop %p (seeds) | mean | changed % |
|---|---|---|---|---|---|---|---|---|
| Spikformer(DE) | off | 91.27 | +0.00 / +0.00 / +0.00 | +0.00 | 0.00 | — | — | — |
| Spikformer(DE) | on | 91.94 | -26.94 / -25.03 / -27.66 | -26.54 | 15.07 | — | — | — |
| base pop LIF | off | 91.51 | -10.63 / -16.36 / -11.82 | -12.94 | 5.62 | -10.63 / -16.36 / -11.82 | -12.94 | 5.62 |
| base pop LIF | on | 91.90 | -24.11 / -33.51 / -26.30 | -27.97 | 15.91 | -3.25 / -2.59 / -6.77 | -4.20 | 2.13 |
| base pop f-LIF | off | 91.41 | -6.91 / -10.30 / -4.93 | -7.38 | 2.95 | -6.91 / -10.30 / -4.93 | -7.38 | 2.95 |
| base pop f-LIF | on | 92.01 | -23.81 / -21.18 / -21.72 | -22.23 | 10.48 | -2.52 / -2.78 / -6.60 | -3.97 | 1.49 |

run time per model: ETT 14-531 s, MIT-BIH 41-122 s
