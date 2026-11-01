# Query form with population neurons on MIT-BIH, stage 1 — summary

See the module docstring for sources and rules. Test macro-F1 (%), last-epoch weights, seeds 42 / 7 / 2026.

| model | macro-F1 (mean ± sd) | accuracy | F1 N / S / V / F / Q | energy µJ |
|---|---|---|---|---|
| Spikformer-style (spk_ssa) | 92.22 ± 0.21 | 98.55 | 99.24 / 85.28 / 96.04 / 81.38 / 99.17 | 9.55 |
| query population LIF (nsmt4q_pop_lif) | 92.17 ± 0.16 | 98.57 | 99.26 / 86.18 / 96.08 / 80.36 / 98.98 | 11.10 |
| query population f-LIF (nsmt4q_pop_flif) | 92.02 ± 0.17 | 98.54 | 99.23 / 85.70 / 96.14 / 79.97 / 99.04 | 12.26 |
| replace population LIF (nsmt4_pop_lif) | 91.56 ± 0.08 | 98.41 | 99.16 / 83.37 / 95.91 / 80.43 / 98.96 | 10.01 |
| replace population f-LIF (nsmt4_pop_flif) | 91.47 ± 0.39 | 98.36 | 99.13 / 82.81 / 95.85 / 80.77 / 98.82 | 11.02 |
| B form population LIF (spkneo_pop_lif) | 92.20 ± 0.20 | 98.57 | 99.25 / 85.96 / 96.22 / 80.52 / 99.06 | 12.37 |
| B form population f-LIF (spkneo_pop_flif) | 92.06 ± 0.13 | 98.56 | 99.25 / 85.41 / 96.25 / 80.34 / 99.05 | 13.49 |

## M1 (main) query population − Spikformer-style

| comparison | A | B | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|---|---|
| M1 query pop_lif − Spikformer-style | q_pop_lif | spk | 92.17 | 92.22 | -0.23 / -0.24 / +0.32 | -0.05 | no clear difference | 1.16 | not met |
| M1 query pop_flif − Spikformer-style | q_pop_flif | spk | 92.02 | 92.22 | -0.38 / -0.41 / +0.17 | -0.20 | no clear difference | 1.28 | not met |

## M2 query − replace ([NSMT T=4]), M3 query − append (B form), report

| comparison | A | B | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B |
|---|---|---|---|---|---|---|---|---|
| M2 query − replace, pop_lif | q_pop_lif | r_pop_lif | 92.17 | 91.56 | +0.72 / +0.38 / +0.73 | +0.61 | better | 1.11 |
| M2 query − replace, pop_flif | q_pop_flif | r_pop_flif | 92.02 | 91.47 | +0.49 / +0.01 / +1.12 | +0.54 | better | 1.11 |
| M3 query − B form, pop_lif | q_pop_lif | b_pop_lif | 92.17 | 92.20 | -0.18 / +0.02 / +0.06 | -0.03 | no clear difference | 0.90 |
| M3 query − B form, pop_flif | q_pop_flif | b_pop_flif | 92.02 | 92.06 | +0.12 / -0.34 / +0.10 | -0.04 | no clear difference | 0.91 |
| report: query population f-LIF − query population LIF | q_pop_flif | q_pop_lif | 92.02 | 92.17 | -0.15 / -0.16 / -0.15 | -0.16 | no clear difference | 1.10 |

**Stage 2 trigger (neither query population better than Spikformer-style in M1):** yes (run stage 2)

