# Aux-loss test (NSMT reconstruction loss on the msn base; spikingjelly and SpikeDE bases) — summary

See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. New runs at sha d9a3e35ab.

## Results (mean ± sd of seeds)

| basis | model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | MIT-BIH macro-F1 % |
|---|---|---|---|---|---|---|
| sj | Spikformer-style | 0.2903 ± 0.0012 | 0.4223 ± 0.0024 | 0.1775 ± 0.0007 | 0.4034 ± 0.0016 | 92.22 ± 0.21 |
| sj | query pop LIF base, aux off | 0.2931 ± 0.0045 | 0.4157 ± 0.0011 | 0.1776 ± 0.0004 | 0.4036 ± 0.0005 | 92.16 ± 0.05 |
| sj | query pop LIF base, aux on | 0.2927 ± 0.0040 | 0.4218 ± 0.0078 | 0.1766 ± 0.0011 | 0.4023 ± 0.0012 | 92.06 ± 0.33 |
| sj | query pop f-LIF base, aux off | 0.2904 ± 0.0039 | 0.4166 ± 0.0055 | 0.1782 ± 0.0011 | 0.4036 ± 0.0031 | 92.27 ± 0.43 |
| sj | query pop f-LIF base, aux on | 0.2899 ± 0.0048 | 0.4224 ± 0.0024 | 0.1773 ± 0.0009 | 0.4030 ± 0.0014 | 91.92 ± 0.10 |
| sde | Spikformer-style | 0.2921 ± 0.0038 | 0.4229 ± 0.0062 | 0.1779 ± 0.0004 | 0.4048 ± 0.0023 | 91.94 ± 0.49 |
| sde | query pop LIF base, aux off | 0.2914 ± 0.0021 | 0.4163 ± 0.0051 | 0.1763 ± 0.0009 | 0.4029 ± 0.0019 | 91.90 ± 0.26 |
| sde | query pop LIF base, aux on | 0.2915 ± 0.0052 | 0.4216 ± 0.0027 | 0.1769 ± 0.0006 | 0.4036 ± 0.0015 | 91.97 ± 0.29 |
| sde | query pop f-LIF base, aux off | 0.2897 ± 0.0037 | 0.4181 ± 0.0065 | 0.1766 ± 0.0013 | 0.4034 ± 0.0020 | 92.01 ± 0.49 |
| sde | query pop f-LIF base, aux on | 0.2913 ± 0.0070 | 0.4168 ± 0.0036 | 0.1773 ± 0.0004 | 0.4043 ± 0.0015 | 91.89 ± 0.09 |

## Basis sj (spikingjelly hippocampus): ETT (r = A / B − 1 of test MSE, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|---|
| A1 query pop LIF: aux on − off | -0.15 (no clear difference) | +1.47 (worse) | -0.57 (no clear difference) | -0.33 (no clear difference) | +0.11 | 6/12 | no clear difference | 1.04 | — |
| A1 query pop f-LIF: aux on − off | -0.20 (no clear difference) | +1.40 (worse) | -0.53 (no clear difference) | -0.15 (no clear difference) | +0.13 | 6/12 | no clear difference | 1.03 | — |
| A2 query pop LIF: base aux − Spikformer-style | +0.81 (no clear difference) | -0.12 (no clear difference) | -0.50 (no clear difference) | -0.29 (no clear difference) | -0.03 | 8/12 | no clear difference | 1.30 | — |
| A2 query pop f-LIF: base aux − Spikformer-style | -0.16 (no clear difference) | +0.02 (no clear difference) | -0.11 (no clear difference) | -0.12 (no clear difference) | -0.09 | 7/12 | no clear difference | 1.56 | — |
| (reference) query pop LIF: base aux off − Spikformer-style | +0.96 (no clear difference) | -1.57 (better) | +0.08 (no clear difference) | +0.04 (no clear difference) | -0.12 | 6/12 | no clear difference | 1.25 | — |
| (reference) query pop f-LIF: base aux off − Spikformer-style | +0.04 (no clear difference) | -1.35 (better) | +0.43 (no clear difference) | +0.03 (no clear difference) | -0.21 | 6/12 | no clear difference | 1.51 | — |

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean g | pairs negative | verdict | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|
| A3 query pop LIF gap change (negative = aux moves the base ahead) | -0.15 | +1.45 | -0.57 | -0.33 | +0.10 | 6/12 | no clear change | — |
| A3 query pop f-LIF gap change (negative = aux moves the base ahead) | -0.20 | +1.37 | -0.54 | -0.15 | +0.12 | 6/12 | no clear change | — |

### Basis sj: MIT-BIH (d = A − B of test macro-F1, %p per seed 42 / 7 / 2026)

| comparison | d per seed | mean | verdict | energy A/B |
|---|---|---|---|---|
| A1 query pop LIF: aux on − off | +0.30 / -0.45 / -0.12 | -0.09 | no clear difference | 1.01 |
| A1 query pop f-LIF: aux on − off | -0.22 / -0.02 / -0.78 | -0.34 | no clear difference | 1.02 |
| A2 query pop LIF: base aux − Spikformer-style | -0.02 / -0.50 / +0.05 | -0.16 | no clear difference | 1.20 |
| A2 query pop f-LIF: base aux − Spikformer-style | -0.61 / -0.26 / -0.03 | -0.30 | no clear difference | 1.34 |
| (reference) query pop LIF: base aux off − Spikformer-style | -0.32 / -0.05 / +0.17 | -0.07 | no clear difference | 1.19 |
| (reference) query pop f-LIF: base aux off − Spikformer-style | -0.38 / -0.24 / +0.76 | +0.04 | no clear difference | 1.32 |
| A3 query pop LIF gap change (positive = aux moves the base ahead) | +0.30 / -0.45 / -0.12 | -0.09 | no clear change | |
| A3 query pop f-LIF gap change (positive = aux moves the base ahead) | -0.22 / -0.02 / -0.78 | -0.34 | no clear change | |

## Basis sde (every neuron SpikeDE): ETT (r = A / B − 1 of test MSE, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|---|
| A1 query pop LIF: aux on − off | +0.04 (no clear difference) | +1.27 (worse) | +0.37 (no clear difference) | +0.16 (no clear difference) | +0.46 | 4/12 | no clear difference | 1.04 | — |
| A1 query pop f-LIF: aux on − off | +0.54 (no clear difference) | -0.30 (no clear difference) | +0.41 (no clear difference) | +0.20 (no clear difference) | +0.21 | 4/12 | no clear difference | 1.03 | — |
| A2 query pop LIF: base aux − Spikformer-style | -0.19 (no clear difference) | -0.29 (no clear difference) | -0.59 (no clear difference) | -0.30 (no clear difference) | -0.34 | 8/12 | no clear difference | 1.29 | — |
| A2 query pop f-LIF: base aux − Spikformer-style | -0.26 (no clear difference) | -1.42 (better) | -0.34 (no clear difference) | -0.13 (no clear difference) | -0.54 | 9/12 | no clear difference | 1.56 | — |
| (reference) query pop LIF: base aux off − Spikformer-style | -0.23 (no clear difference) | -1.55 (better) | -0.95 (no clear difference) | -0.46 (no clear difference) | -0.80 | 11/12 | no clear difference | 1.25 | — |
| (reference) query pop f-LIF: base aux off − Spikformer-style | -0.80 (no clear difference) | -1.12 (better) | -0.74 (no clear difference) | -0.33 (no clear difference) | -0.75 | 9/12 | no clear difference | 1.51 | — |

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean g | pairs negative | verdict | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|
| A3 query pop LIF gap change (negative = aux moves the base ahead) | +0.04 | +1.26 | +0.36 | +0.16 | +0.46 | 4/12 | no clear change | — |
| A3 query pop f-LIF gap change (negative = aux moves the base ahead) | +0.54 | -0.30 | +0.40 | +0.20 | +0.21 | 4/12 | no clear change | — |

### Basis sde: MIT-BIH (d = A − B of test macro-F1, %p per seed 42 / 7 / 2026)

| comparison | d per seed | mean | verdict | energy A/B |
|---|---|---|---|---|
| A1 query pop LIF: aux on − off | +0.30 / +0.30 / -0.40 | +0.07 | no clear difference | 1.01 |
| A1 query pop f-LIF: aux on − off | +0.51 / -0.46 / -0.40 | -0.12 | no clear difference | 1.00 |
| A2 query pop LIF: base aux − Spikformer-style | +0.15 / -0.22 / +0.14 | +0.03 | no clear difference | 1.22 |
| A2 query pop f-LIF: base aux − Spikformer-style | +0.21 / -0.58 / +0.22 | -0.05 | no clear difference | 1.35 |
| (reference) query pop LIF: base aux off − Spikformer-style | -0.14 / -0.52 / +0.55 | -0.04 | no clear difference | 1.21 |
| (reference) query pop f-LIF: base aux off − Spikformer-style | -0.30 / -0.12 / +0.62 | +0.07 | no clear difference | 1.35 |
| A3 query pop LIF gap change (positive = aux moves the base ahead) | +0.30 / +0.30 / -0.40 | +0.07 | no clear change | |
| A3 query pop f-LIF gap change (positive = aux moves the base ahead) | +0.51 / -0.46 / -0.40 | -0.12 | no clear change | |

## conclusions (rule of the pre-registration)

| basis | comparison | task | pop LIF | pop f-LIF | conclusion over the two neurons |
|---|---|---|---|---|---|
| sj | A1 aux on − off | ETT | no clear difference | no clear difference | no clear difference |
| sj | A1 aux on − off | MIT-BIH | no clear difference | no clear difference | no clear difference |
| sj | A2 base aux − Spikformer-style | ETT | no clear difference | no clear difference | no clear difference |
| sj | A2 base aux − Spikformer-style | MIT-BIH | no clear difference | no clear difference | no clear difference |
| sj | A3 gap change | ETT | no clear change | no clear change | no clear change |
| sj | A3 gap change | MIT-BIH | no clear change | no clear change | no clear change |
| sj | overall: base aux better than Spikformer-style | both | not supported | not supported | — |
| sde | A1 aux on − off | ETT | no clear difference | no clear difference | no clear difference |
| sde | A1 aux on − off | MIT-BIH | no clear difference | no clear difference | no clear difference |
| sde | A2 base aux − Spikformer-style | ETT | no clear difference | no clear difference | no clear difference |
| sde | A2 base aux − Spikformer-style | MIT-BIH | no clear difference | no clear difference | no clear difference |
| sde | A3 gap change | ETT | no clear change | no clear change | no clear change |
| sde | A3 gap change | MIT-BIH | no clear change | no clear change | no clear change |
| sde | overall: base aux better than Spikformer-style | both | weak (direction only) | not supported | — |

## report only

| basis | model | ETT energy µJ / Spikformer-style | MIT-BIH energy µJ / Spikformer-style | ETT rho_neo | MIT-BIH rho_neo | train reconstruction MSE, last epoch (ETT / MIT-BIH) | ETT best validation at epoch 0 (of 12) | ETT epochs (min–max) | ETT run time s (min–max) |
|---|---|---|---|---|---|---|---|---|---|
| sj | query pop LIF aux off | 13.04 / 1.25 | 11.37 / 1.19 | 0.0328 | 0.0587 | — / 0.0955 | 1 | 11–22 | 172–1179 |
| sj | query pop LIF aux on | 13.61 / 1.30 | 11.50 / 1.20 | 0.1426 | 0.0666 | 0.1687 / 0.0077 | 1 | 11–22 | 180–4312 |
| sj | query pop f-LIF aux off | 15.86 / 1.51 | 12.57 / 1.32 | 0.0371 | 0.0757 | — / 0.0972 | 2 | 11–20 | 177–1039 |
| sj | query pop f-LIF aux on | 16.35 / 1.56 | 12.77 / 1.34 | 0.1407 | 0.0854 | 0.0859 / 0.0036 | 3 | 11–22 | 176–3252 |
| sde | query pop LIF aux off | 13.14 / 1.25 | 11.42 / 1.21 | 0.0332 | 0.0718 | — / 0.0984 | 3 | 11–18 | 190–2103 |
| sde | query pop LIF aux on | 13.63 / 1.29 | 11.50 / 1.22 | 0.1296 | 0.0745 | 0.1763 / 0.0072 | 4 | 11–18 | 175–3798 |
| sde | query pop f-LIF aux off | 15.93 / 1.51 | 12.77 / 1.35 | 0.0323 | 0.0967 | — / 0.1066 | 4 | 11–18 | 182–2095 |
| sde | query pop f-LIF aux on | 16.43 / 1.56 | 12.71 / 1.35 | 0.1360 | 0.0943 | 0.0877 / 0.0035 | 3 | 11–17 | 195–2486 |

### coupling diagnostics of the aux arms (spikingjelly basis; means of seeds): ΔMSE % when the Neocortex / position / content is removed at test time

| model | cell | varying share | ΔMSE Neocortex 0 | ΔMSE position 0 | ΔMSE content 0 | y rate |
|---|---|---|---|---|---|---|
| nsmt4q_pop_lif_msn_aux | ETTh2 H96 | 0.700 | -1.07 | -0.17 | -1.07 | 0.0200 |
| nsmt4q_pop_lif_msn_aux | ETTh2 H720 | 0.554 | -0.55 | +0.26 | -0.55 | 0.0206 |
| nsmt4q_pop_lif_msn_aux | ETTm2 H96 | 0.197 | +3.68 | +2.27 | +3.84 | 0.0640 |
| nsmt4q_pop_lif_msn_aux | ETTm2 H720 | 0.334 | +2.01 | +0.26 | +2.40 | 0.0304 |
| nsmt4q_pop_flif_msn_aux | ETTh2 H96 | 0.682 | +0.09 | -0.22 | +0.09 | 0.0231 |
| nsmt4q_pop_flif_msn_aux | ETTh2 H720 | 0.562 | -0.15 | +0.39 | -0.15 | 0.0302 |
| nsmt4q_pop_flif_msn_aux | ETTm2 H96 | 0.189 | +1.58 | +1.83 | +1.85 | 0.0646 |
| nsmt4q_pop_flif_msn_aux | ETTm2 H720 | 0.395 | +1.23 | +0.20 | +1.25 | 0.0259 |
