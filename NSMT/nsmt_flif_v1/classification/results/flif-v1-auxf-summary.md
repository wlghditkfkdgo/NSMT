# Full-signal aux test (raw patches as the reconstruction target; spikingjelly and SpikeDE bases) — summary

See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. New runs at sha 9c1262613.

## Results (mean ± sd of seeds)

| basis | model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | MIT-BIH macro-F1 % |
|---|---|---|---|---|---|---|
| sj | Spikformer-style | 0.2903 ± 0.0012 | 0.4223 ± 0.0024 | 0.1775 ± 0.0007 | 0.4034 ± 0.0016 | 92.22 ± 0.21 |
| sj | query pop LIF base, aux off | 0.2931 ± 0.0045 | 0.4157 ± 0.0011 | 0.1776 ± 0.0004 | 0.4036 ± 0.0005 | 92.16 ± 0.05 |
| sj | query pop LIF base, low-pass aux | 0.2927 ± 0.0040 | 0.4218 ± 0.0078 | 0.1766 ± 0.0011 | 0.4023 ± 0.0012 | 92.06 ± 0.33 |
| sj | query pop LIF base, full-signal aux | 0.2915 ± 0.0052 | 0.4179 ± 0.0020 | 0.1775 ± 0.0011 | 0.4031 ± 0.0026 | 92.26 ± 0.11 |
| sj | query pop f-LIF base, aux off | 0.2904 ± 0.0039 | 0.4166 ± 0.0055 | 0.1782 ± 0.0011 | 0.4036 ± 0.0031 | 92.27 ± 0.43 |
| sj | query pop f-LIF base, low-pass aux | 0.2899 ± 0.0048 | 0.4224 ± 0.0024 | 0.1773 ± 0.0009 | 0.4030 ± 0.0014 | 91.92 ± 0.10 |
| sj | query pop f-LIF base, full-signal aux | 0.2900 ± 0.0030 | 0.4193 ± 0.0078 | 0.1774 ± 0.0004 | 0.4032 ± 0.0022 | 92.08 ± 0.17 |
| sde | Spikformer-style | 0.2921 ± 0.0038 | 0.4229 ± 0.0062 | 0.1779 ± 0.0004 | 0.4048 ± 0.0023 | 91.94 ± 0.49 |
| sde | query pop LIF base, aux off | 0.2914 ± 0.0021 | 0.4163 ± 0.0051 | 0.1763 ± 0.0009 | 0.4029 ± 0.0019 | 91.90 ± 0.26 |
| sde | query pop LIF base, low-pass aux | 0.2915 ± 0.0052 | 0.4216 ± 0.0027 | 0.1769 ± 0.0006 | 0.4036 ± 0.0015 | 91.97 ± 0.29 |
| sde | query pop LIF base, full-signal aux | 0.2916 ± 0.0016 | 0.4202 ± 0.0053 | 0.1762 ± 0.0007 | 0.4023 ± 0.0009 | 91.81 ± 0.37 |
| sde | query pop f-LIF base, aux off | 0.2897 ± 0.0037 | 0.4181 ± 0.0065 | 0.1766 ± 0.0013 | 0.4034 ± 0.0020 | 92.01 ± 0.49 |
| sde | query pop f-LIF base, low-pass aux | 0.2913 ± 0.0070 | 0.4168 ± 0.0036 | 0.1773 ± 0.0004 | 0.4043 ± 0.0015 | 91.89 ± 0.09 |
| sde | query pop f-LIF base, full-signal aux | 0.2927 ± 0.0044 | 0.4209 ± 0.0039 | 0.1767 ± 0.0007 | 0.4040 ± 0.0017 | 91.75 ± 0.43 |

## Basis sj (spikingjelly hippocampus): ETT (r = A / B − 1 of test MSE, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|---|
| F1 query pop LIF: full − off | -0.54 (no clear difference) | +0.53 (no clear difference) | -0.05 (no clear difference) | -0.13 (no clear difference) | -0.05 | 5/12 | no clear difference | 1.05 | — |
| F1 query pop f-LIF: full − off | -0.14 (no clear difference) | +0.65 (no clear difference) | -0.49 (no clear difference) | -0.08 (no clear difference) | -0.02 | 7/12 | no clear difference | 1.03 | — |
| F2 query pop LIF: full − low-pass | -0.39 (no clear difference) | -0.91 (no clear difference) | +0.53 (no clear difference) | +0.20 (no clear difference) | -0.14 | 7/12 | no clear difference | 1.00 | — |
| F2 query pop f-LIF: full − low-pass | +0.06 (no clear difference) | -0.74 (no clear difference) | +0.05 (no clear difference) | +0.07 (no clear difference) | -0.14 | 6/12 | no clear difference | 1.00 | — |
| F3 query pop LIF: base full − Spikformer-style | +0.42 (no clear difference) | -1.05 (better) | +0.03 (no clear difference) | -0.09 (no clear difference) | -0.17 | 7/12 | no clear difference | 1.30 | — |
| F3 query pop f-LIF: base full − Spikformer-style | -0.11 (no clear difference) | -0.71 (no clear difference) | -0.06 (no clear difference) | -0.05 (no clear difference) | -0.23 | 7/12 | no clear difference | 1.57 | — |

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean g | pairs negative | verdict | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|
| F3 gap change query pop LIF (negative = the full aux moves the base ahead) | -0.55 | +0.53 | -0.05 | -0.13 | -0.05 | 5/12 | no clear change | — |
| F3 gap change query pop f-LIF (negative = the full aux moves the base ahead) | -0.15 | +0.64 | -0.49 | -0.08 | -0.02 | 7/12 | no clear change | — |

### Basis sj: MIT-BIH (d = A − B of test macro-F1, %p per seed 42 / 7 / 2026)

| comparison | d per seed | mean | verdict | energy A/B |
|---|---|---|---|---|
| M1 query pop LIF: full − off | +0.19 / +0.16 / -0.04 | +0.10 | no clear difference | 1.00 |
| M1 query pop f-LIF: full − off | +0.21 / -0.10 / -0.67 | -0.18 | no clear difference | 1.01 |
| M2 query pop LIF: full − low-pass | -0.10 / +0.61 / +0.08 | +0.20 | no clear difference | 0.99 |
| M2 query pop f-LIF: full − low-pass | +0.44 / -0.08 / +0.12 | +0.16 | no clear difference | 0.99 |
| M3 query pop LIF: base full − Spikformer-style | -0.13 / +0.11 / +0.13 | +0.04 | no clear difference | 1.19 |
| M3 gap change query pop LIF (positive = the full aux moves the base ahead) | +0.19 / +0.16 / -0.04 | +0.10 | no clear change | |
| M3 query pop f-LIF: base full − Spikformer-style | -0.17 / -0.34 / +0.09 | -0.14 | no clear difference | 1.33 |
| M3 gap change query pop f-LIF (positive = the full aux moves the base ahead) | +0.21 / -0.10 / -0.67 | -0.18 | no clear change | |

## Basis sde (every neuron SpikeDE): ETT (r = A / B − 1 of test MSE, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|---|
| F1 query pop LIF: full − off | +0.07 (no clear difference) | +0.94 (no clear difference) | -0.00 (no clear difference) | -0.16 (no clear difference) | +0.21 | 5/12 | no clear difference | 1.04 | — |
| F1 query pop f-LIF: full − off | +1.04 (worse) | +0.68 (no clear difference) | +0.04 (no clear difference) | +0.13 (no clear difference) | +0.47 | 4/12 | no clear difference | 1.03 | — |
| F2 query pop LIF: full − low-pass | +0.04 (no clear difference) | -0.33 (no clear difference) | -0.37 (no clear difference) | -0.31 (no clear difference) | -0.24 | 9/12 | no clear difference | 1.00 | — |
| F2 query pop f-LIF: full − low-pass | +0.50 (no clear difference) | +0.98 (no clear difference) | -0.37 (no clear difference) | -0.07 (no clear difference) | +0.26 | 6/12 | no clear difference | 1.00 | — |
| F3 query pop LIF: base full − Spikformer-style | -0.16 (no clear difference) | -0.62 (no clear difference) | -0.95 (no clear difference) | -0.61 (no clear difference) | -0.59 | 10/12 | no clear difference | 1.30 | — |
| F3 query pop f-LIF: base full − Spikformer-style | +0.22 (no clear difference) | -0.46 (no clear difference) | -0.71 (no clear difference) | -0.20 (no clear difference) | -0.29 | 8/12 | no clear difference | 1.56 | — |

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean g | pairs negative | verdict | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|
| F3 gap change query pop LIF (negative = the full aux moves the base ahead) | +0.07 | +0.93 | -0.00 | -0.16 | +0.21 | 5/12 | no clear change | — |
| F3 gap change query pop f-LIF (negative = the full aux moves the base ahead) | +1.02 | +0.66 | +0.04 | +0.13 | +0.46 | 4/12 | no clear change | — |

### Basis sde: MIT-BIH (d = A − B of test macro-F1, %p per seed 42 / 7 / 2026)

| comparison | d per seed | mean | verdict | energy A/B |
|---|---|---|---|---|
| M1 query pop LIF: full − off | -0.05 / +0.26 / -0.48 | -0.09 | no clear difference | 1.01 |
| M1 query pop f-LIF: full − off | +0.58 / -0.42 / -0.94 | -0.26 | no clear difference | 0.99 |
| M2 query pop LIF: full − low-pass | -0.34 / -0.05 / -0.07 | -0.16 | no clear difference | 1.00 |
| M2 query pop f-LIF: full − low-pass | +0.07 / +0.04 / -0.54 | -0.14 | no clear difference | 0.99 |
| M3 query pop LIF: base full − Spikformer-style | -0.19 / -0.27 / +0.07 | -0.13 | no clear difference | 1.22 |
| M3 gap change query pop LIF (positive = the full aux moves the base ahead) | -0.05 / +0.26 / -0.48 | -0.09 | no clear change | |
| M3 query pop f-LIF: base full − Spikformer-style | +0.28 / -0.54 / -0.32 | -0.19 | no clear difference | 1.33 |
| M3 gap change query pop f-LIF (positive = the full aux moves the base ahead) | +0.58 / -0.42 / -0.94 | -0.26 | no clear change | |

## conclusions (rule of the pre-registration)

| basis | comparison | task | pop LIF | pop f-LIF | conclusion over the two neurons |
|---|---|---|---|---|---|
| sj | full − off | ETT | no clear difference | no clear difference | no clear difference |
| sj | full − off | MIT-BIH | no clear difference | no clear difference | no clear difference |
| sj | full − low-pass | ETT | no clear difference | no clear difference | no clear difference |
| sj | full − low-pass | MIT-BIH | no clear difference | no clear difference | no clear difference |
| sj | base full − Spikformer-style | ETT | no clear difference | no clear difference | no clear difference |
| sj | base full − Spikformer-style | MIT-BIH | no clear difference | no clear difference | no clear difference |
| sj | gap change | ETT | no clear change | no clear change | no clear change |
| sj | gap change | MIT-BIH | no clear change | no clear change | no clear change |
| sj | overall: base full aux better than Spikformer-style | both | weak (direction only) | not supported | — |
| sde | full − off | ETT | no clear difference | no clear difference | no clear difference |
| sde | full − off | MIT-BIH | no clear difference | no clear difference | no clear difference |
| sde | full − low-pass | ETT | no clear difference | no clear difference | no clear difference |
| sde | full − low-pass | MIT-BIH | no clear difference | no clear difference | no clear difference |
| sde | base full − Spikformer-style | ETT | no clear difference | no clear difference | no clear difference |
| sde | base full − Spikformer-style | MIT-BIH | no clear difference | no clear difference | no clear difference |
| sde | gap change | ETT | no clear change | no clear change | no clear change |
| sde | gap change | MIT-BIH | no clear change | no clear change | no clear change |
| sde | overall: base full aux better than Spikformer-style | both | not supported | not supported | — |

## report only

| basis | model | ETT energy µJ / Spikformer-style | MIT-BIH energy µJ / Spikformer-style | ETT rho_neo | MIT-BIH rho_neo | train reconstruction MSE, last epoch (ETT / MIT-BIH) | ETT best validation at epoch 0 (of 12) | ETT epochs (min–max) |
|---|---|---|---|---|---|---|---|---|
| sj | query pop LIF off | 13.04 / 1.25 | 11.37 / 1.19 | 0.0328 | 0.0587 | — / 0.0955 | 1 | 11–22 |
| sj | query pop LIF low | 13.61 / 1.30 | 11.50 / 1.20 | 0.1426 | 0.0666 | 0.1687 / 0.0077 | 1 | 11–22 |
| sj | query pop LIF full | 13.65 / 1.30 | 11.39 / 1.19 | 0.1491 | 0.0602 | 0.2818 / 0.0109 | 3 | 11–18 |
| sj | query pop f-LIF off | 15.86 / 1.51 | 12.57 / 1.32 | 0.0371 | 0.0757 | — / 0.0972 | 2 | 11–20 |
| sj | query pop f-LIF low | 16.35 / 1.56 | 12.77 / 1.34 | 0.1407 | 0.0854 | 0.0859 / 0.0036 | 3 | 11–22 |
| sj | query pop f-LIF full | 16.42 / 1.57 | 12.68 / 1.33 | 0.1542 | 0.0761 | 0.1444 / 0.0060 | 3 | 11–26 |
| sde | query pop LIF off | 13.14 / 1.25 | 11.42 / 1.21 | 0.0332 | 0.0718 | — / 0.0984 | 3 | 11–18 |
| sde | query pop LIF low | 13.63 / 1.29 | 11.50 / 1.22 | 0.1296 | 0.0745 | 0.1763 / 0.0072 | 4 | 11–18 |
| sde | query pop LIF full | 13.69 / 1.30 | 11.51 / 1.22 | 0.1378 | 0.0800 | 0.2900 / 0.0106 | 3 | 11–17 |
| sde | query pop f-LIF off | 15.93 / 1.51 | 12.77 / 1.35 | 0.0323 | 0.0967 | — / 0.1066 | 4 | 11–18 |
| sde | query pop f-LIF low | 16.43 / 1.56 | 12.71 / 1.35 | 0.1360 | 0.0943 | 0.0877 / 0.0035 | 3 | 11–17 |
| sde | query pop f-LIF full | 16.48 / 1.56 | 12.58 / 1.33 | 0.1456 | 0.0872 | 0.1484 / 0.0053 | 4 | 11–17 |

### coupling diagnostics of the full-signal aux arms (spikingjelly basis; means of seeds)

| model | cell | varying share | ΔMSE Neocortex 0 | ΔMSE position 0 | ΔMSE content 0 | y rate |
|---|---|---|---|---|---|---|
| nsmt4q_pop_lif_msn_auxf | ETTh2 H96 | 0.620 | -0.88 | -0.22 | -0.87 | 0.0210 |
| nsmt4q_pop_lif_msn_auxf | ETTh2 H720 | 0.521 | -0.05 | +0.15 | -0.05 | 0.0211 |
| nsmt4q_pop_lif_msn_auxf | ETTm2 H96 | 0.187 | +2.91 | +2.36 | +2.75 | 0.0709 |
| nsmt4q_pop_lif_msn_auxf | ETTm2 H720 | 0.396 | +1.69 | +0.16 | +1.69 | 0.0271 |
| nsmt4q_pop_flif_msn_auxf | ETTh2 H96 | 0.646 | -0.00 | -0.13 | -0.00 | 0.0217 |
| nsmt4q_pop_flif_msn_auxf | ETTh2 H720 | 0.580 | +0.40 | +0.19 | +0.40 | 0.0237 |
| nsmt4q_pop_flif_msn_auxf | ETTm2 H96 | 0.188 | +1.03 | +2.87 | +1.14 | 0.0702 |
| nsmt4q_pop_flif_msn_auxf | ETTm2 H720 | 0.327 | +1.47 | +0.36 | +1.53 | 0.0237 |
