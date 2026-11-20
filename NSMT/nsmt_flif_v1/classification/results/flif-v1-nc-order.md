# 4-2 models: test-time token-order shuffle with a random order per row (stored models, no training; report only)

See the docstring of order_eval_nc.py. (나) is the pre-registered target (d53119ef4); the base (range 1) and (가) rows are added for reference. R = 5 repeats (generator seed 1000 + r). Run shas: 546d16ba5, d53119ef4.

- body: tokens shuffled for both paths, the hippocampus output put back before the head (does the body use the order?)
- neo: only the Neocortex reads the tokens in the random order, its outputs put back
- all: as body, not put back (ETT: the flat head also reads the tokens at the wrong positions)

## ETT (r = shuffled test MSE / normal test MSE - 1, %; mean of the R repeats per run, then over the 12 runs = 4 cells x 3 seeds)

| neuron | model | normal MSE (mean of 12) | body r % (min .. max of 12) | runs body r > 0 | neo r % (min .. max) | all r % | largest repeat sd of body r |
|---|---|---|---|---|---|---|---|
| pop LIF | base | 0.3222 | +0.66 (-1.93 .. +4.03) | 7/12 | +0.66 (-1.93 .. +4.03) | +21.4 | 0.05 |
| pop LIF | (가) Neocortex current + attention | 0.3232 | +1.82 (-1.23 .. +8.81) | 9/12 | +1.82 (-1.23 .. +8.81) | +21.5 | 0.11 |
| pop LIF | (나) Neocortex current, no attention | 0.3232 | +1.88 (-1.38 .. +6.21) | 10/12 | +1.88 (-1.38 .. +6.21) | +21.5 | 0.15 |
| pop f-LIF | base | 0.3228 | +0.59 (-1.15 .. +3.08) | 10/12 | +0.59 (-1.15 .. +3.08) | +21.1 | 0.06 |
| pop f-LIF | (가) Neocortex current + attention | 0.3248 | +1.12 (-0.99 .. +3.72) | 8/12 | +1.12 (-0.99 .. +3.72) | +21.0 | 0.09 |
| pop f-LIF | (나) Neocortex current, no attention | 0.3237 | +1.42 (-1.07 .. +4.16) | 10/12 | +1.42 (-1.07 .. +4.16) | +21.4 | 0.09 |

### ETT body r % per cell (mean of 3 seeds; seeds 42 / 7 / 2026 in brackets)

| neuron | model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|---|
| pop LIF | base | -1.48 (-0.70 / -1.93 / -1.80) | -0.00 (-0.45 / +0.79 / -0.35) | +3.24 (+4.03 / +1.95 / +3.75) | +0.88 (+1.17 / +1.20 / +0.29) |
| pop LIF | (가) Neocortex current + attention | -0.93 (-0.32 / -1.23 / -1.23) | +0.66 (+0.68 / +1.10 / +0.20) | +6.54 (+5.64 / +5.16 / +8.81) | +1.02 (+0.27 / +2.19 / +0.60) |
| pop LIF | (나) Neocortex current, no attention | -0.55 (+0.72 / -1.38 / -0.99) | +1.41 (+0.63 / +1.10 / +2.50) | +5.65 (+4.81 / +5.92 / +6.21) | +1.01 (+0.41 / +2.06 / +0.57) |
| pop f-LIF | base | -0.03 (+0.57 / -1.15 / +0.48) | +1.57 (+1.67 / +3.08 / -0.03) | +0.73 (+1.05 / +1.14 / +0.00) | +0.09 (+0.01 / +0.25 / +0.02) |
| pop f-LIF | (가) Neocortex current + attention | -0.38 (-0.99 / -0.11 / -0.05) | +0.20 (+0.13 / +1.14 / -0.66) | +3.50 (+3.24 / +3.72 / +3.55) | +1.14 (+0.15 / +2.28 / +1.00) |
| pop f-LIF | (나) Neocortex current, no attention | -0.38 (-0.42 / +0.35 / -1.07) | +0.58 (+0.27 / +0.96 / +0.50) | +3.96 (+3.62 / +4.11 / +4.16) | +1.51 (+0.17 / +3.00 / +1.35) |

## MIT-BIH (macro-F1 %; drop = shuffled - normal, %p, mean of the R repeats per seed; changed = share of test beats whose prediction changes, %)

| neuron | model | normal | body drop %p (seed 42 / 7 / 2026) | mean | changed % | neo drop %p (seeds) | mean | changed % |
|---|---|---|---|---|---|---|---|---|
| pop LIF | base | 91.39 | -13.64 / -8.16 / -10.85 | -10.88 | 4.86 | -13.64 / -8.16 / -10.85 | -10.88 | 4.86 |
| pop LIF | (가) Neocortex current + attention | 91.68 | -53.86 / -51.57 / -53.59 | -53.01 | 36.01 | -53.86 / -51.57 / -53.59 | -53.01 | 36.01 |
| pop LIF | (나) Neocortex current, no attention | 91.46 | -52.41 / -50.23 / -53.57 | -52.07 | 39.32 | -52.41 / -50.23 / -53.57 | -52.07 | 39.32 |
| pop f-LIF | base | 91.57 | -4.63 / -6.87 / -4.75 | -5.42 | 2.58 | -4.63 / -6.87 / -4.75 | -5.42 | 2.58 |
| pop f-LIF | (가) Neocortex current + attention | 91.17 | -42.91 / -42.42 / -42.73 | -42.69 | 26.38 | -42.91 / -42.42 / -42.73 | -42.69 | 26.38 |
| pop f-LIF | (나) Neocortex current, no attention | 91.02 | -41.87 / -43.71 / -44.94 | -43.51 | 26.39 | -41.87 / -43.71 / -44.94 | -43.51 | 26.39 |

run time per model: ETT 25-203 s, MIT-BIH 50-75 s
