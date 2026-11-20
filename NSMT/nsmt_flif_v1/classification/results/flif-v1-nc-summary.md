# 4-2 coupling placement ((가) Neocortex current + attention, (나) Neocortex current without attention) — summary

See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. New runs at sha d53119ef4; reference runs (range 1) at sha 546d16ba5.

## Results (mean ± sd of seeds; ETT test MSE, MIT-BIH test macro-F1 %)

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | MIT-BIH macro-F1 % |
|---|---|---|---|---|---|
| Spikformer(DE) | 0.2903 ± 0.0032 | 0.4249 ± 0.0087 | 0.1767 ± 0.0002 | 0.4048 ± 0.0015 | 91.20 ± 0.29 |
| query pop LIF base | 0.2902 ± 0.0018 | 0.4196 ± 0.0057 | 0.1756 ± 0.0007 | 0.4034 ± 0.0007 | 91.39 ± 0.41 |
| query pop LIF (가) Neocortex current + attention | 0.2917 ± 0.0040 | 0.4221 ± 0.0064 | 0.1765 ± 0.0012 | 0.4027 ± 0.0022 | 91.68 ± 0.14 |
| query pop LIF (나) Neocortex current, no attention | 0.2909 ± 0.0043 | 0.4235 ± 0.0066 | 0.1761 ± 0.0011 | 0.4023 ± 0.0019 | 91.46 ± 0.31 |
| query pop f-LIF base | 0.2897 ± 0.0047 | 0.4207 ± 0.0074 | 0.1765 ± 0.0009 | 0.4044 ± 0.0012 | 91.57 ± 0.14 |
| query pop f-LIF (가) Neocortex current + attention | 0.2942 ± 0.0047 | 0.4240 ± 0.0022 | 0.1768 ± 0.0003 | 0.4040 ± 0.0007 | 91.17 ± 0.09 |
| query pop f-LIF (나) Neocortex current, no attention | 0.2926 ± 0.0031 | 0.4224 ± 0.0045 | 0.1766 ± 0.0008 | 0.4030 ± 0.0011 | 91.02 ± 0.11 |

## ETT (r = A / B − 1 of test MSE, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|---|
| G1 query pop LIF: (가) Neocortex current + attention − base | +0.50 (no clear difference) | +0.60 (no clear difference) | +0.49 (no clear difference) | -0.18 (no clear difference) | +0.35 | 4/12 | no clear difference | 1.02 | — |
| G1 query pop f-LIF: (가) Neocortex current + attention − base | +1.56 (worse) | +0.80 (no clear difference) | +0.18 (no clear difference) | -0.10 (no clear difference) | +0.61 | 4/12 | no clear difference | 1.02 | — |
| G2 query pop LIF: (나) Neocortex current, no attention − base | +0.22 (no clear difference) | +0.94 (no clear difference) | +0.28 (no clear difference) | -0.28 (no clear difference) | +0.29 | 7/12 | no clear difference | 0.64 | — |
| G2 query pop f-LIF: (나) Neocortex current, no attention − base | +1.02 (worse) | +0.42 (no clear difference) | +0.10 (no clear difference) | -0.36 (no clear difference) | +0.30 | 6/12 | no clear difference | 0.71 | — |
| G3 query pop LIF: (가) Neocortex current + attention − Spikformer(DE) | +0.49 (no clear difference) | -0.66 (no clear difference) | -0.12 (no clear difference) | -0.52 (no clear difference) | -0.20 | 7/12 | no clear difference | 1.26 | — |
| G3 query pop f-LIF: (가) Neocortex current + attention − Spikformer(DE) | +1.35 (worse) | -0.19 (no clear difference) | +0.05 (no clear difference) | -0.20 (no clear difference) | +0.25 | 5/12 | no clear difference | 1.53 | — |
| G4 query pop LIF: (나) Neocortex current, no attention − Spikformer(DE) | +0.21 (no clear difference) | -0.32 (no clear difference) | -0.33 (no clear difference) | -0.62 (no clear difference) | -0.26 | 9/12 | no clear difference | 0.79 | — |
| G4 query pop f-LIF: (나) Neocortex current, no attention − Spikformer(DE) | +0.82 (no clear difference) | -0.57 (no clear difference) | -0.03 (no clear difference) | -0.46 (no clear difference) | -0.06 | 8/12 | no clear difference | 1.06 | — |

## MIT-BIH (d = A − B of test macro-F1, %p per seed 42 / 7 / 2026)

| comparison | d per seed | mean | verdict | energy A/B |
|---|---|---|---|---|
| G1 query pop LIF: (가) Neocortex current + attention − base | +0.03 / +0.83 / +0.00 | +0.29 | no clear difference | 1.17 |
| G1 query pop f-LIF: (가) Neocortex current + attention − base | -0.40 / -0.47 / -0.31 | -0.39 | no clear difference | 1.09 |
| G2 query pop LIF: (나) Neocortex current, no attention − base | -0.28 / +0.44 / +0.07 | +0.08 | no clear difference | 0.62 |
| G2 query pop f-LIF: (나) Neocortex current, no attention − base | -0.53 / -0.61 / -0.49 | -0.54 | worse | 0.63 |
| G3 query pop LIF: (가) Neocortex current + attention − Spikformer(DE) | +0.07 / +0.50 / +0.86 | +0.48 | no clear difference | 1.46 |
| G3 query pop f-LIF: (가) Neocortex current + attention − Spikformer(DE) | -0.18 / -0.11 / +0.21 | -0.03 | no clear difference | 1.51 |
| G4 query pop LIF: (나) Neocortex current, no attention − Spikformer(DE) | -0.24 / +0.12 / +0.92 | +0.26 | no clear difference | 0.77 |
| G4 query pop f-LIF: (나) Neocortex current, no attention − Spikformer(DE) | -0.31 / -0.25 / +0.03 | -0.18 | no clear difference | 0.87 |

## conclusions (rules of the pre-registration)

| comparison | task | pop LIF | pop f-LIF | conclusion over the two neurons |
|---|---|---|---|---|
| G1 (가) − base | ETT | no clear difference | no clear difference | no clear difference |
| G1 (가) − base | MIT-BIH | no clear difference | no clear difference | no clear difference |
| G2 (나) − base | ETT | no clear difference | no clear difference | no clear difference |
| G2 (나) − base | MIT-BIH | no clear difference | worse | neuron-dependent (weak) |
| G3 (가) − Spikformer(DE) | ETT | no clear difference | no clear difference | no clear difference |
| G3 (가) − Spikformer(DE) | MIT-BIH | no clear difference | no clear difference | no clear difference |
| G4 (나) − Spikformer(DE) | ETT | no clear difference | no clear difference | no clear difference |
| G4 (나) − Spikformer(DE) | MIT-BIH | no clear difference | no clear difference | no clear difference |
| overall: (가) better than Spikformer(DE) | both | weak (direction only) | not supported | — |
| overall: (나) better than Spikformer(DE) | both | weak (direction only) | not supported | — |

**Recommendation (pre-registered rule; the structure decision is the user's):** keep the base: G2 has a "worse" and G1 has no "better" without a "worse". G2 verdicts (ETT pop LIF, ETT pop f-LIF, MIT-BIH pop LIF, MIT-BIH pop f-LIF): no clear difference, no clear difference, no clear difference, worse; G1: no clear difference, no clear difference, no clear difference, no clear difference.

## report only

| model | ETT energy µJ / Spikformer(DE) | MIT-BIH energy µJ / Spikformer(DE) | ETT rho_hippo | ETT rho_neo | MIT-BIH rho_hippo | MIT-BIH rho_neo | MIT-BIH accuracy % | ETT best at epoch 0 (of 12) | ETT epochs (min–max) | ETT best at an inner validation (of 12) |
|---|---|---|---|---|---|---|---|---|---|---|
| Spikformer(DE) | 10.65 / 1.00 | 9.18 / 1.00 | 0.0610 | — | 0.1075 | — | 98.32 | 2 | 11–17 | 9 |
| query pop LIF base | 13.20 / 1.24 | 11.45 / 1.25 | 0.0600 | 0.0347 | 0.1271 | 0.0785 | 98.40 | 2 | 11–16 | 10 |
| query pop LIF (가) Neocortex current + attention | 13.45 / 1.26 | 13.38 / 1.46 | 0.0598 | 0.0461 | 0.1210 | 0.1360 | 98.47 | 4 | 11–15 | 9 |
| query pop LIF (나) Neocortex current, no attention | 8.48 / 0.79 | 7.10 / 0.77 | 0.0816 | 0.0469 | 0.1618 | 0.1229 | 98.41 | 4 | 11–15 | 10 |
| query pop f-LIF base | 16.00 / 1.50 | 12.65 / 1.38 | 0.0597 | 0.0346 | 0.1253 | 0.0955 | 98.42 | 2 | 11–17 | 11 |
| query pop f-LIF (가) Neocortex current + attention | 16.34 / 1.53 | 13.83 / 1.51 | 0.0612 | 0.0497 | 0.1132 | 0.1113 | 98.36 | 4 | 11–15 | 10 |
| query pop f-LIF (나) Neocortex current, no attention | 11.35 / 1.06 | 8.02 / 0.87 | 0.0847 | 0.0495 | 0.1492 | 0.1100 | 98.29 | 4 | 11–16 | 10 |

| model | ETT alpha mean | ETT mean abs alpha | ETT share alpha < 0 | MIT-BIH alpha mean | MIT-BIH mean abs alpha | MIT-BIH share alpha < 0 |
|---|---|---|---|---|---|---|
| query pop LIF (가) Neocortex current + attention | +1.000 | +1.000 | 0.00 | +1.427 | +1.427 | 0.00 |
| query pop LIF (나) Neocortex current, no attention | +1.002 | +1.002 | 0.00 | +1.543 | +1.543 | 0.00 |
| query pop f-LIF (가) Neocortex current + attention | +1.002 | +1.002 | 0.00 | +1.519 | +1.519 | 0.00 |
| query pop f-LIF (나) Neocortex current, no attention | +1.006 | +1.006 | 0.00 | +1.630 | +1.630 | 0.00 |

Uncounted alpha addition (syops counts modules; alpha * spikes is not a module): upper bound with every Neocortex output unit firing at every token, C x T_h x N x D additions x 0.9 pJ.

| task | C x T_h x N x D | bound µJ | share of the smallest (나) energy |
|---|---|---|---|
| ETT (7 channels, 24 tokens, D 64) | 43008 | 0.0387 | 0.46 % |
| MIT-BIH (11 tokens, D 384) | 16896 | 0.0152 | 0.21 % |
