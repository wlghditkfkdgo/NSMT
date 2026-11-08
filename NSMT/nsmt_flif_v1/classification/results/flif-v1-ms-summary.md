# Membrane shortcut test (query population LIF / f-LIF) — summary

See the module docstring and docs/MODI_PROJECT_LOG.md 2026-11-08 for the pre-registered rules.

## ETT — test MSE (mean ± sd of seeds 42 / 7 / 2026, best-validation weights)

| model | variant | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean validation L1 |
|---|---|---|---|---|---|---|
| query pop LIF | base (IAND, no Neocortex residual) | 0.2912 ± 0.0045 | 0.4135 ± 0.0014 | 0.1760 ± 0.0002 | 0.4028 ± 0.0022 | 0.3618 |
| query pop LIF | msh (hippocampus MS) | 0.2919 ± 0.0027 | 0.4173 ± 0.0041 | 0.1759 ± 0.0010 | 0.4011 ± 0.0023 | 0.3632 |
| query pop LIF | msn (Neocortex MS) | 0.2931 ± 0.0045 | 0.4157 ± 0.0011 | 0.1776 ± 0.0004 | 0.4036 ± 0.0005 | 0.3624 |
| query pop LIF | mshn (both) | 0.2916 ± 0.0023 | 0.4219 ± 0.0083 | 0.1775 ± 0.0004 | 0.4032 ± 0.0034 | 0.3631 |
| query pop f-LIF | base (IAND, no Neocortex residual) | 0.2890 ± 0.0049 | 0.4135 ± 0.0018 | 0.1765 ± 0.0010 | 0.4019 ± 0.0014 | 0.3616 |
| query pop f-LIF | msh (hippocampus MS) | 0.2945 ± 0.0049 | 0.4191 ± 0.0042 | 0.1759 ± 0.0011 | 0.4027 ± 0.0036 | 0.3628 |
| query pop f-LIF | msn (Neocortex MS) | 0.2904 ± 0.0039 | 0.4166 ± 0.0055 | 0.1782 ± 0.0011 | 0.4036 ± 0.0031 | 0.3618 |
| query pop f-LIF | mshn (both) | 0.2960 ± 0.0049 | 0.4182 ± 0.0043 | 0.1782 ± 0.0017 | 0.4036 ± 0.0019 | 0.3629 |

## ETT — verdicts (r = A / B − 1 of test MSE, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B |
|---|---|---|---|---|---|---|---|---|
| M1 query pop LIF: msh − base (request 2: hippocampus MS) | +0.26 (no clear difference) | +0.92 (no clear difference) | -0.08 (no clear difference) | -0.42 (no clear difference) | +0.17 | 8/12 | no clear difference | 1.04 |
| M1 query pop f-LIF: msh − base (request 2: hippocampus MS) | +1.92 (worse) | +1.34 (worse) | -0.38 (no clear difference) | +0.18 (no clear difference) | +0.77 | 4/12 | no clear difference | 1.03 |
| M2 query pop LIF: msn − base (request 3 alone: Neocortex MS) | +0.66 (no clear difference) | +0.52 (no clear difference) | +0.89 (no clear difference) | +0.20 (no clear difference) | +0.57 | 2/12 | no clear difference | 1.01 |
| M2 query pop f-LIF: msn − base (request 3 alone: Neocortex MS) | +0.48 (no clear difference) | +0.74 (no clear difference) | +0.96 (no clear difference) | +0.41 (no clear difference) | +0.65 | 2/12 | no clear difference | 1.01 |
| M3 query pop LIF: mshn − msh (request 3 on top of request 2) | -0.08 (no clear difference) | +1.11 (no clear difference) | +0.91 (no clear difference) | +0.51 (no clear difference) | +0.61 | 4/12 | no clear difference | 1.01 |
| M3 query pop f-LIF: mshn − msh (request 3 on top of request 2) | +0.50 (no clear difference) | -0.21 (no clear difference) | +1.35 (worse) | +0.23 (no clear difference) | +0.47 | 4/12 | no clear difference | 1.01 |
| report query pop LIF: mshn − base (both vs base (report only)) | +0.16 (no clear difference) | +2.02 (worse) | +0.83 (no clear difference) | +0.09 (no clear difference) | +0.78 | 4/12 | no clear difference | 1.04 |
| report query pop f-LIF: mshn − base (both vs base (report only)) | +2.43 (worse) | +1.12 (worse) | +0.97 (no clear difference) | +0.41 (no clear difference) | +1.23 | 2/12 | worse | 1.04 |

## MIT-BIH — test macro-F1 (%, last-epoch weights, lr 0.01)

| model | variant | seed 42 | seed 7 | seed 2026 | mean | accuracy | F1 N / S / V / F / Q |
|---|---|---|---|---|---|---|---|
| query pop LIF | base (IAND, no Neocortex residual) | 92.19 | 92.00 | 92.33 | 92.17 | 98.57 | 99.3 / 86.2 / 96.1 / 80.4 / 99.0 |
| query pop LIF | msh (hippocampus MS) | 91.79 | 91.54 | 91.50 | 91.61 | 98.43 | 99.2 / 84.7 / 96.0 / 79.4 / 98.8 |
| query pop LIF | msn (Neocortex MS) | 92.10 | 92.20 | 92.17 | 92.16 | 98.56 | 99.2 / 85.8 / 96.0 / 80.7 / 99.0 |
| query pop LIF | mshn (both) | 92.19 | 92.07 | 91.64 | 91.97 | 98.45 | 99.2 / 84.7 / 95.9 / 81.2 / 98.8 |
| query pop f-LIF | base (IAND, no Neocortex residual) | 92.04 | 91.84 | 92.18 | 92.02 | 98.54 | 99.2 / 85.7 / 96.1 / 80.0 / 99.0 |
| query pop f-LIF | msh (hippocampus MS) | 91.71 | 91.89 | 92.16 | 91.92 | 98.50 | 99.2 / 84.7 / 96.0 / 80.7 / 98.9 |
| query pop f-LIF | msn (Neocortex MS) | 92.04 | 92.00 | 92.76 | 92.27 | 98.57 | 99.3 / 85.7 / 96.1 / 81.2 / 99.1 |
| query pop f-LIF | mshn (both) | 92.11 | 91.47 | 91.99 | 91.86 | 98.47 | 99.2 / 85.1 / 96.0 / 80.1 / 98.8 |

## MIT-BIH — verdicts (Δ = A − B, %p)

| comparison | per seed (42 / 7 / 2026) | mean | > 0 | verdict | energy A/B |
|---|---|---|---|---|---|
| M1 query pop LIF: msh − base (request 2: hippocampus MS) | -0.40 / -0.46 / -0.83 | -0.56 | 0/3 | worse | 0.80 |
| M1 query pop f-LIF: msh − base (request 2: hippocampus MS) | -0.33 / +0.05 / -0.01 | -0.10 | 1/3 | no clear difference | 0.82 |
| M2 query pop LIF: msn − base (request 3 alone: Neocortex MS) | -0.09 / +0.20 / -0.16 | -0.02 | 1/3 | no clear difference | 1.02 |
| M2 query pop f-LIF: msn − base (request 3 alone: Neocortex MS) | -0.00 / +0.17 / +0.58 | +0.25 | 2/3 | no clear difference | 1.03 |
| M3 query pop LIF: mshn − msh (request 3 on top of request 2) | +0.40 / +0.53 / +0.15 | +0.36 | 3/3 | no clear difference | 1.05 |
| M3 query pop f-LIF: mshn − msh (request 3 on top of request 2) | +0.41 / -0.42 / -0.17 | -0.06 | 1/3 | no clear difference | 1.04 |
| report query pop LIF: mshn − base (both vs base (report only)) | -0.00 / +0.07 / -0.68 | -0.20 | 1/3 | no clear difference | 0.84 |
| report query pop f-LIF: mshn − base (both vs base (report only)) | +0.07 / -0.37 / -0.18 | -0.16 | 1/3 | no clear difference | 0.86 |

## conclusions (rule of the pre-registration)

| comparison | task | pop LIF | pop f-LIF | conclusion |
|---|---|---|---|---|
| M1 msh − base (request 2: hippocampus MS) | ETT | no clear difference | no clear difference | no clear difference |
| M1 msh − base (request 2: hippocampus MS) | MIT-BIH | worse | no clear difference | neuron-dependent (weak) |
| M2 msn − base (request 3 alone: Neocortex MS) | ETT | no clear difference | no clear difference | no clear difference |
| M2 msn − base (request 3 alone: Neocortex MS) | MIT-BIH | no clear difference | no clear difference | no clear difference |
| M3 mshn − msh (request 3 on top of request 2) | ETT | no clear difference | no clear difference | no clear difference |
| M3 mshn − msh (request 3 on top of request 2) | MIT-BIH | no clear difference | no clear difference | no clear difference |

## report only

### firing rates and energy (mean of seeds; ETT: mean over the 4 cells)

| model | variant | ETT rho_neo | ETT rho_hippo | ETT energy uJ (head recount) | ETT MS additions uJ | MIT-BIH rho_neo | MIT-BIH rho_hippo | MIT-BIH energy uJ (logged) | MIT-BIH MS additions uJ | params ETTh2 H96 / MIT-BIH |
|---|---|---|---|---|---|---|---|---|---|---|
| query pop LIF | base | 0.0190 | 0.0531 | 12.94 | 0.0000 | 0.0424 | 0.1476 | 11.10 | 0.0000 | 175544 / 1230888 |
| query pop LIF | msh | 0.0219 | 0.0635 | 13.41 | 0.0774 | 0.0462 | 0.0990 | 8.83 | 0.0304 | 175544 / 1230888 |
| query pop LIF | msn | 0.0328 | 0.0574 | 13.04 | 0.0145 | 0.0587 | 0.1506 | 11.37 | 0.0076 | 175544 / 1230888 |
| query pop LIF | mshn | 0.0363 | 0.0652 | 13.54 | 0.0919 | 0.0700 | 0.1057 | 9.28 | 0.0380 | 175544 / 1230888 |
| query pop f-LIF | base | 0.0196 | 0.0541 | 15.74 | 0.0000 | 0.0623 | 0.1398 | 12.26 | 0.0000 | 175544 / 1230888 |
| query pop f-LIF | msh | 0.0202 | 0.0619 | 16.15 | 0.0774 | 0.0628 | 0.0992 | 10.09 | 0.0304 | 175544 / 1230888 |
| query pop f-LIF | msn | 0.0371 | 0.0581 | 15.86 | 0.0145 | 0.0757 | 0.1476 | 12.57 | 0.0076 | 175544 / 1230888 |
| query pop f-LIF | mshn | 0.0402 | 0.0663 | 16.35 | 0.0919 | 0.0828 | 0.1036 | 10.51 | 0.0380 | 175544 / 1230888 |

### test MAE (mean of seeds)

| model | variant | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|---|
| query pop LIF | base | 0.3347 | 0.4322 | 0.2529 | 0.3956 |
| query pop LIF | msh | 0.3354 | 0.4357 | 0.2530 | 0.3946 |
| query pop LIF | msn | 0.3356 | 0.4330 | 0.2535 | 0.3954 |
| query pop LIF | mshn | 0.3355 | 0.4369 | 0.2538 | 0.3949 |
| query pop f-LIF | base | 0.3343 | 0.4325 | 0.2528 | 0.3952 |
| query pop f-LIF | msh | 0.3371 | 0.4362 | 0.2528 | 0.3950 |
| query pop f-LIF | msn | 0.3347 | 0.4338 | 0.2540 | 0.3956 |
| query pop f-LIF | mshn | 0.3376 | 0.4355 | 0.2546 | 0.3951 |

### every model vs Spikformer-style (flif-v1-gen-ett / flif-v1-gen-mitbih; same rules)

| model | ETT: ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | ETT R % | lower | ETT verdict | ETT energy ratio | MIT-BIH Δ %p | MIT-BIH verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| query pop LIF base | +0.30 | -2.07 | -0.80 | -0.16 | -0.68 | 9/12 | no clear difference | 1.24 | -0.05 | no clear difference |
| query pop LIF msh | +0.53 | -1.18 | -0.88 | -0.58 | -0.53 | 10/12 | no clear difference | 1.28 | -0.61 | worse |
| query pop LIF msn | +0.96 | -1.57 | +0.08 | +0.04 | -0.12 | 6/12 | no clear difference | 1.25 | -0.07 | no clear difference |
| query pop LIF mshn | +0.45 | -0.09 | +0.02 | -0.07 | +0.08 | 7/12 | no clear difference | 1.29 | -0.25 | no clear difference |
| query pop f-LIF base | -0.44 | -2.08 | -0.53 | -0.37 | -0.85 | 9/12 | no clear difference | 1.50 | -0.20 | no clear difference |
| query pop f-LIF msh | +1.43 | -0.77 | -0.91 | -0.19 | -0.11 | 9/12 | no clear difference | 1.54 | -0.30 | no clear difference |
| query pop f-LIF msn | +0.04 | -1.35 | +0.43 | +0.03 | -0.21 | 6/12 | no clear difference | 1.51 | +0.04 | no clear difference |
| query pop f-LIF mshn | +1.94 | -0.98 | +0.43 | +0.03 | +0.36 | 6/12 | no clear difference | 1.56 | -0.36 | no clear difference |
