# Query form with population neurons on MIT-BIH, stage 2 (learning rate) — summary

See the module docstring for sources and rules. All runs train on 80 % of the training file (--mitbih_val); validation = the
held-out 20 % (last epoch, selection only); test = the 30 % test file (last-epoch weights). Default learning rate marked *.

## every learning rate: validation macro-F1 / test macro-F1 (mean over seeds, %)

| model | 0.0025 | 0.005 | 0.01* | 0.02 | selected |
|---|---|---|---|---|---|
| query population LIF | 91.77 / 91.09 | 92.16 / 91.22 | 92.01 / 91.50 | 92.28 / 91.66 | 0.02 |
| query population f-LIF | 91.85 / 91.15 | 92.07 / 91.56 | 92.49 / 91.52 | 92.41 / 91.41 | 0.01 |
| Spikformer-style | 91.93 / 91.17 | 92.60 / 91.31 | 92.43 / 91.76 | 92.29 / 91.57 | 0.005 |

## energy µJ (mean over seeds)

| model | 0.0025 | 0.005 | 0.01 | 0.02 |
|---|---|---|---|---|
| query population LIF | 10.70 | 11.10 | 11.12 | 11.02 |
| query population f-LIF | 11.82 | 12.16 | 12.28 | 12.10 |
| Spikformer-style | 9.27 | 9.48 | 9.51 | 9.27 |

## T1 selected − default learning rate (test)

| comparison | A | B | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B |
|---|---|---|---|---|---|---|---|---|
| T1 query population LIF @0.02 − @0.01 | q_pop_lif@0.02 | q_pop_lif@0.01 | 91.66 | 91.50 | +0.32 / -0.64 / +0.78 | +0.15 | no clear difference | 0.99 |
| T1 Spikformer-style @0.005 − @0.01 | spk@0.005 | spk@0.01 | 91.31 | 91.76 | -0.13 / -0.98 / -0.23 | -0.45 | no clear difference | 1.00 |

## T2 query population − Spikformer-style, each at its selected learning rate (main) and at the default (report)

| comparison | A | B | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|---|---|
| T2 query population LIF @0.02 − Spikformer-style @0.005 | q_pop_lif@0.02 | spk@0.005 | 91.66 | 91.31 | +0.76 / -0.03 / +0.32 | +0.35 | no clear difference | 1.16 | not met |
| T2 query population f-LIF @0.01 − Spikformer-style @0.005 | q_pop_flif@0.01 | spk@0.005 | 91.52 | 91.31 | +0.48 / -0.00 / +0.18 | +0.22 | no clear difference | 1.30 | not met |
| report: query population LIF − Spikformer-style, both @0.01 | q_pop_lif@0.01 | spk@0.01 | 91.50 | 91.76 | +0.31 / -0.37 / -0.70 | -0.25 | no clear difference | 1.17 | not met |
| report: query population f-LIF − Spikformer-style, both @0.01 | q_pop_flif@0.01 | spk@0.01 | 91.52 | 91.76 | +0.35 / -0.98 / -0.06 | -0.23 | no clear difference | 1.29 | not met |
