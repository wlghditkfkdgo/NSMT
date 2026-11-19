# Positional-encoding-off test (relative position bias of the attention removed, retrained; every neuron SpikeDE) — summary

See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. New runs at sha 96fb4dc2a.

## Results (mean ± sd of seeds; ETT test MSE, MIT-BIH test macro-F1 %)

| model | bias | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | MIT-BIH macro-F1 % |
|---|---|---|---|---|---|---|
| Spikformer(DE) | on | 0.2921 ± 0.0038 | 0.4229 ± 0.0062 | 0.1779 ± 0.0004 | 0.4048 ± 0.0023 | 91.94 ± 0.49 |
| Spikformer(DE) | off | 0.2886 ± 0.0025 | 0.4179 ± 0.0011 | 0.1769 ± 0.0011 | 0.4044 ± 0.0025 | 91.27 ± 0.16 |
| query pop LIF base (SpikeDE) | on | 0.2914 ± 0.0021 | 0.4163 ± 0.0051 | 0.1763 ± 0.0009 | 0.4029 ± 0.0019 | 91.90 ± 0.26 |
| query pop LIF base (SpikeDE) | off | 0.2899 ± 0.0030 | 0.4164 ± 0.0044 | 0.1770 ± 0.0008 | 0.4028 ± 0.0016 | 91.51 ± 0.28 |
| query pop f-LIF base (SpikeDE) | on | 0.2897 ± 0.0037 | 0.4181 ± 0.0065 | 0.1766 ± 0.0013 | 0.4034 ± 0.0020 | 92.01 ± 0.49 |
| query pop f-LIF base (SpikeDE) | off | 0.2917 ± 0.0033 | 0.4166 ± 0.0022 | 0.1764 ± 0.0008 | 0.4027 ± 0.0018 | 91.41 ± 0.46 |

## ETT (r = A / B − 1 of test MSE, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|---|
| N1 Spikformer(DE): off − on | -1.18 (better) | -1.16 (better) | -0.60 (no clear difference) | -0.11 (no clear difference) | -0.76 | 9/12 | no clear difference | 1.00 | — |
| N1 query pop LIF base (SpikeDE): off − on | -0.52 (no clear difference) | +0.02 (no clear difference) | +0.40 (no clear difference) | -0.05 (no clear difference) | -0.04 | 8/12 | no clear difference | 1.00 | — |
| N1 query pop f-LIF base (SpikeDE): off − on | +0.66 (no clear difference) | -0.34 (no clear difference) | -0.12 (no clear difference) | -0.19 (no clear difference) | +0.00 | 6/12 | no clear difference | 1.00 | — |
| N2 query pop LIF: base (off) − Spikformer(DE) (off) | +0.43 (no clear difference) | -0.36 (no clear difference) | +0.04 (no clear difference) | -0.39 (no clear difference) | -0.07 | 6/12 | no clear difference | 1.24 | — |
| (reference) query pop LIF: base − Spikformer(DE), bias on | -0.23 (no clear difference) | -1.55 (better) | -0.95 (no clear difference) | -0.46 (no clear difference) | -0.80 | 11/12 | no clear difference | 1.25 | — |
| N2 query pop f-LIF: base (off) − Spikformer(DE) (off) | +1.04 (worse) | -0.30 (no clear difference) | -0.27 (no clear difference) | -0.41 (no clear difference) | +0.02 | 5/12 | no clear difference | 1.51 | — |
| (reference) query pop f-LIF: base − Spikformer(DE), bias on | -0.80 (no clear difference) | -1.12 (better) | -0.74 (no clear difference) | -0.33 (no clear difference) | -0.75 | 9/12 | no clear difference | 1.51 | — |

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean g | pairs negative | verdict | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|
| N3 gap change query pop LIF (negative = the base gains when the bias is removed) | +0.66 | +1.19 | +1.00 | +0.06 | +0.73 | 4/12 | no clear change | — |
| N3 gap change query pop f-LIF (negative = the base gains when the bias is removed) | +1.84 | +0.82 | +0.47 | -0.08 | +0.77 | 3/12 | no clear change | — |

## MIT-BIH (d = A − B of test macro-F1, %p per seed 42 / 7 / 2026)

| comparison | d per seed | mean | verdict | energy A/B |
|---|---|---|---|---|
| N1 Spikformer(DE): off − on | -0.32 / -1.39 / -0.29 | -0.67 | worse | 0.98 |
| N1 query pop LIF base (SpikeDE): off − on | -0.28 / -0.62 / -0.29 | -0.39 | no clear difference | 0.99 |
| N1 query pop f-LIF base (SpikeDE): off − on | +0.04 / -0.56 / -1.28 | -0.60 | worse | 0.99 |
| N2 query pop LIF: base (off) − Spikformer(DE) (off) | -0.10 / +0.25 / +0.55 | +0.24 | no clear difference | 1.23 |
| (reference) query pop LIF: base − Spikformer(DE), bias on | -0.14 / -0.52 / +0.55 | -0.04 | no clear difference | 1.21 |
| N3 gap change query pop LIF (positive = the base gains when the bias is removed) | +0.05 / +0.77 / +0.01 | +0.28 | no clear change | |
| N2 query pop f-LIF: base (off) − Spikformer(DE) (off) | +0.06 / +0.72 / -0.36 | +0.14 | no clear difference | 1.37 |
| (reference) query pop f-LIF: base − Spikformer(DE), bias on | -0.30 / -0.12 / +0.62 | +0.07 | no clear difference | 1.35 |
| N3 gap change query pop f-LIF (positive = the base gains when the bias is removed) | +0.36 / +0.84 / -0.99 | +0.07 | no clear change | |

## conclusions (rule of the pre-registration)

| comparison | task | Spikformer(DE) | pop LIF | pop f-LIF | conclusion over the two neurons |
|---|---|---|---|---|---|
| N1 off − on | ETT | no clear difference | no clear difference | no clear difference | (per model) |
| N1 off − on | MIT-BIH | worse | no clear difference | worse | (per model) |
| N2 base − Spikformer(DE), bias off | ETT | — | no clear difference | no clear difference | no clear difference |
| N2 base − Spikformer(DE), bias off | MIT-BIH | — | no clear difference | no clear difference | no clear difference |
| N3 gap change | ETT | — | no clear change | no clear change | no clear change |
| N3 gap change | MIT-BIH | — | no clear change | no clear change | no clear change |
| overall: base better than Spikformer(DE), bias off | both | — | weak (direction only) | not supported | — |

## report only

| model | bias | ETT energy µJ / Spikformer(DE) same arm | MIT-BIH energy µJ / Spikformer(DE) same arm | ETT rho_hippo | ETT rho_neo | MIT-BIH rho_hippo | MIT-BIH rho_neo | MIT-BIH accuracy % | ETT best validation at epoch 0 (of 12) | ETT epochs (min–max) |
|---|---|---|---|---|---|---|---|---|---|---|
| Spikformer(DE) | on | 10.55 / 1.00 | 9.45 / 1.00 | 0.0608 | — | 0.1165 | — | 98.53 | 2 | 11–16 |
| Spikformer(DE) | off | 10.54 / 1.00 | 9.26 / 1.00 | 0.0588 | — | 0.1092 | — | 98.33 | 4 | 11–23 |
| query pop LIF base (SpikeDE) | on | 13.14 / 1.25 | 11.42 / 1.21 | 0.0585 | 0.0332 | 0.1236 | 0.0718 | 98.47 | 3 | 11–18 |
| query pop LIF base (SpikeDE) | off | 13.11 / 1.24 | 11.35 / 1.23 | 0.0571 | 0.0311 | 0.1269 | 0.0707 | 98.41 | 4 | 11–18 |
| query pop f-LIF base (SpikeDE) | on | 15.93 / 1.51 | 12.77 / 1.35 | 0.0584 | 0.0323 | 0.1240 | 0.0967 | 98.54 | 4 | 11–18 |
| query pop f-LIF base (SpikeDE) | off | 15.90 / 1.51 | 12.67 / 1.37 | 0.0576 | 0.0314 | 0.1233 | 0.0977 | 98.39 | 3 | 11–22 |
