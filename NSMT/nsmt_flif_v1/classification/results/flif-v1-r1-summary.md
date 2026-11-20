# Hyperparameter range 1 (BN / bias without weight decay; ETT four validations per epoch) — summary

See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. New runs at sha 546d16ba5.

## Results (mean ± sd of seeds; ETT test MSE, MIT-BIH test macro-F1 %)

| model | training | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | MIT-BIH macro-F1 % |
|---|---|---|---|---|---|---|
| Spikformer(DE) | old | 0.2886 ± 0.0025 | 0.4179 ± 0.0011 | 0.1769 ± 0.0011 | 0.4044 ± 0.0025 | 91.27 ± 0.16 |
| Spikformer(DE) | new | 0.2903 ± 0.0032 | 0.4249 ± 0.0087 | 0.1767 ± 0.0002 | 0.4048 ± 0.0015 | 91.20 ± 0.29 |
| query pop LIF base (SpikeDE) | old | 0.2899 ± 0.0030 | 0.4164 ± 0.0044 | 0.1770 ± 0.0008 | 0.4028 ± 0.0016 | 91.51 ± 0.28 |
| query pop LIF base (SpikeDE) | new | 0.2902 ± 0.0018 | 0.4196 ± 0.0057 | 0.1756 ± 0.0007 | 0.4034 ± 0.0007 | 91.39 ± 0.41 |
| query pop f-LIF base (SpikeDE) | old | 0.2917 ± 0.0033 | 0.4166 ± 0.0022 | 0.1764 ± 0.0008 | 0.4027 ± 0.0018 | 91.41 ± 0.46 |
| query pop f-LIF base (SpikeDE) | new | 0.2897 ± 0.0047 | 0.4207 ± 0.0074 | 0.1765 ± 0.0009 | 0.4044 ± 0.0012 | 91.57 ± 0.14 |

## ETT (r = A / B − 1 of test MSE, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|---|
| R1 Spikformer(DE): new − old | +0.58 (no clear difference) | +1.68 (worse) | -0.12 (no clear difference) | +0.12 (no clear difference) | +0.57 | 6/12 | no clear difference | 1.01 | — |
| R1 query pop LIF base (SpikeDE): new − old | +0.12 (no clear difference) | +0.79 (no clear difference) | -0.76 (no clear difference) | +0.17 (no clear difference) | +0.08 | 7/12 | no clear difference | 1.01 | — |
| R1 query pop f-LIF base (SpikeDE): new − old | -0.68 (no clear difference) | +0.98 (no clear difference) | +0.03 (no clear difference) | +0.43 (no clear difference) | +0.19 | 7/12 | no clear difference | 1.01 | — |
| R2 query pop LIF: base − Spikformer(DE), new training | -0.02 (no clear difference) | -1.25 (better) | -0.60 (no clear difference) | -0.34 (no clear difference) | -0.55 | 9/12 | no clear difference | 1.24 | — |
| (reference) query pop LIF: base − Spikformer(DE), old training (bias-off N2) | +0.43 (no clear difference) | -0.36 (no clear difference) | +0.04 (no clear difference) | -0.39 (no clear difference) | -0.07 | 6/12 | no clear difference | 1.24 | — |
| R2 query pop f-LIF: base − Spikformer(DE), new training | -0.19 (no clear difference) | -0.99 (no clear difference) | -0.12 (no clear difference) | -0.10 (no clear difference) | -0.35 | 9/12 | no clear difference | 1.50 | — |
| (reference) query pop f-LIF: base − Spikformer(DE), old training (bias-off N2) | +1.04 (worse) | -0.30 (no clear difference) | -0.27 (no clear difference) | -0.41 (no clear difference) | +0.02 | 5/12 | no clear difference | 1.51 | — |

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean g | pairs negative | verdict | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|
| R3 gap change query pop LIF (negative = the base gains with the new training) | -0.44 | -0.88 | -0.65 | +0.05 | -0.48 | 8/12 | no clear change | — |
| R3 gap change query pop f-LIF (negative = the base gains with the new training) | -1.23 | -0.69 | +0.15 | +0.31 | -0.37 | 7/12 | no clear change | — |

## MIT-BIH (d = A − B of test macro-F1, %p per seed 42 / 7 / 2026)

| comparison | d per seed | mean | verdict | energy A/B |
|---|---|---|---|---|
| R1 Spikformer(DE): new − old | +0.02 / +0.15 / -0.39 | -0.07 | no clear difference | 0.99 |
| R1 query pop LIF base (SpikeDE): new − old | +0.15 / -0.43 / -0.09 | -0.12 | no clear difference | 1.01 |
| R1 query pop f-LIF base (SpikeDE): new − old | +0.18 / -0.21 / +0.50 | +0.16 | no clear difference | 1.00 |
| R2 query pop LIF: base − Spikformer(DE), new training | +0.03 / -0.33 / +0.85 | +0.18 | no clear difference | 1.25 |
| (reference) query pop LIF: base − Spikformer(DE), old training | -0.10 / +0.25 / +0.55 | +0.24 | no clear difference | 1.23 |
| R3 gap change query pop LIF (positive = the base gains with the new training) | +0.13 / -0.58 / +0.30 | -0.05 | no clear change | |
| R2 query pop f-LIF: base − Spikformer(DE), new training | +0.22 / +0.36 / +0.52 | +0.37 | no clear difference | 1.38 |
| (reference) query pop f-LIF: base − Spikformer(DE), old training | +0.06 / +0.72 / -0.36 | +0.14 | no clear difference | 1.37 |
| R3 gap change query pop f-LIF (positive = the base gains with the new training) | +0.16 / -0.36 / +0.89 | +0.23 | no clear change | |

## conclusions (rule of the pre-registration)

| comparison | task | Spikformer(DE) | pop LIF | pop f-LIF | conclusion over the two neurons |
|---|---|---|---|---|---|
| R1 new − old | ETT | no clear difference | no clear difference | no clear difference | (per model) |
| R1 new − old | MIT-BIH | no clear difference | no clear difference | no clear difference | (per model) |
| R2 base − Spikformer(DE), new training | ETT | — | no clear difference | no clear difference | no clear difference |
| R2 base − Spikformer(DE), new training | MIT-BIH | — | no clear difference | no clear difference | no clear difference |
| R3 gap change | ETT | — | no clear change | no clear change | no clear change |
| R3 gap change | MIT-BIH | — | no clear change | no clear change | no clear change |
| overall: base better than Spikformer(DE), new training | both | — | weak (direction only) | weak (direction only) | — |

## report only

| model | training | ETT energy µJ / Spikformer(DE) same arm | MIT-BIH energy µJ / Spikformer(DE) same arm | ETT rho_hippo | ETT rho_neo | MIT-BIH rho_hippo | MIT-BIH rho_neo | MIT-BIH accuracy % | ETT best at epoch 0 (of 12) | ETT epochs (min–max) | ETT best at an inner validation (of 12) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Spikformer(DE) | old | 10.54 / 1.00 | 9.26 / 1.00 | 0.0588 | — | 0.1092 | — | 98.33 | 4 | 11–23 | — |
| Spikformer(DE) | new | 10.65 / 1.00 | 9.18 / 1.00 | 0.0610 | — | 0.1075 | — | 98.32 | 2 | 11–17 | 9 |
| query pop LIF base (SpikeDE) | old | 13.11 / 1.24 | 11.35 / 1.23 | 0.0571 | 0.0311 | 0.1269 | 0.0707 | 98.41 | 4 | 11–18 | — |
| query pop LIF base (SpikeDE) | new | 13.20 / 1.24 | 11.45 / 1.25 | 0.0600 | 0.0347 | 0.1271 | 0.0785 | 98.40 | 2 | 11–16 | 10 |
| query pop f-LIF base (SpikeDE) | old | 15.90 / 1.51 | 12.67 / 1.37 | 0.0576 | 0.0314 | 0.1233 | 0.0977 | 98.39 | 3 | 11–22 | — |
| query pop f-LIF base (SpikeDE) | new | 16.00 / 1.50 | 12.65 / 1.38 | 0.0597 | 0.0346 | 0.1253 | 0.0955 | 98.42 | 2 | 11–17 | 11 |
