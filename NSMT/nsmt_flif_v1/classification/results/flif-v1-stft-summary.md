# STFT test (Neocortex-MS query population LIF / f-LIF; (ga) soma-less tau 2 vs (na) soma tau 4-32) — summary

See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules.

## ETT — test MSE (mean ± sd of seeds 42 / 7 / 2026, best-validation weights)

| model | variant | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean validation L1 |
|---|---|---|---|---|---|---|
| query pop LIF | msn (base) | 0.2931 ± 0.0045 | 0.4157 ± 0.0011 | 0.1776 ± 0.0004 | 0.4036 ± 0.0005 | 0.3624 |
| query pop LIF | stft = (na): soma, tau 4-32, STFT bands | 0.2918 ± 0.0032 | 0.4194 ± 0.0081 | 0.1767 ± 0.0006 | 0.4031 ± 0.0020 | 0.3606 |
| query pop LIF | fire_stft = (ga): no soma, tau 2, STFT bands | 0.2887 ± 0.0016 | 0.4178 ± 0.0049 | 0.1777 ± 0.0014 | 0.4040 ± 0.0026 | 0.3613 |
| query pop LIF | fire_same: no soma, tau 2, unsplit x4 | 0.2907 ± 0.0026 | 0.4170 ± 0.0051 | 0.1765 ± 0.0003 | 0.4040 ± 0.0021 | 0.3618 |
| query pop LIF | freq (11-13): soma, tau 4-32, DCT bands | 0.2888 ± 0.0023 | 0.4272 ± 0.0154 | 0.1761 ± 0.0005 | 0.4022 ± 0.0019 | 0.3602 |
| query pop LIF | same (11-13): soma, tau 4-32, unsplit x4 | 0.2941 ± 0.0041 | 0.4181 ± 0.0026 | 0.1776 ± 0.0016 | 0.4026 ± 0.0024 | 0.3619 |
| query pop f-LIF | msn (base) | 0.2904 ± 0.0039 | 0.4166 ± 0.0055 | 0.1782 ± 0.0011 | 0.4036 ± 0.0031 | 0.3618 |
| query pop f-LIF | stft = (na): soma, tau 4-32, STFT bands | 0.2920 ± 0.0017 | 0.4195 ± 0.0102 | 0.1764 ± 0.0010 | 0.4032 ± 0.0019 | 0.3605 |
| query pop f-LIF | fire_stft = (ga): no soma, tau 2, STFT bands | 0.2873 ± 0.0038 | 0.4158 ± 0.0035 | 0.1773 ± 0.0014 | 0.4039 ± 0.0030 | 0.3618 |
| query pop f-LIF | fire_same: no soma, tau 2, unsplit x4 | 0.2884 ± 0.0034 | 0.4191 ± 0.0081 | 0.1772 ± 0.0003 | 0.4042 ± 0.0020 | 0.3620 |
| query pop f-LIF | freq (11-13): soma, tau 4-32, DCT bands | 0.2912 ± 0.0036 | 0.4251 ± 0.0144 | 0.1758 ± 0.0009 | 0.4025 ± 0.0016 | 0.3603 |
| query pop f-LIF | same (11-13): soma, tau 4-32, unsplit x4 | 0.2874 ± 0.0065 | 0.4192 ± 0.0062 | 0.1781 ± 0.0011 | 0.4031 ± 0.0027 | 0.3614 |

## ETT — verdicts (r = A / B − 1 of test MSE, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B |
|---|---|---|---|---|---|---|---|---|
| S1 query pop LIF: stft − msn ((na) vs the base) | -0.44 (no clear difference) | +0.88 (no clear difference) | -0.49 (no clear difference) | -0.13 (no clear difference) | -0.04 | 7/12 | no clear difference | 1.02 |
| S1 query pop f-LIF: stft − msn ((na) vs the base) | +0.54 (no clear difference) | +0.70 (no clear difference) | -1.00 (no clear difference) | -0.10 (no clear difference) | +0.04 | 9/12 | no clear difference | 1.02 |
| S2 query pop LIF: fire_stft − msn ((ga) vs the base) | -1.48 (better) | +0.50 (no clear difference) | +0.03 (no clear difference) | +0.10 (no clear difference) | -0.21 | 7/12 | no clear difference | 1.00 |
| S2 query pop f-LIF: fire_stft − msn ((ga) vs the base) | -1.06 (better) | -0.19 (no clear difference) | -0.54 (no clear difference) | +0.08 (no clear difference) | -0.43 | 5/12 | no clear difference | 0.92 |
| S3 query pop LIF: fire_stft − stft ((ga) vs (na)) | -1.05 (better) | -0.36 (no clear difference) | +0.52 (no clear difference) | +0.23 (no clear difference) | -0.16 | 6/12 | no clear difference | 0.98 |
| S3 query pop f-LIF: fire_stft − stft ((ga) vs (na)) | -1.59 (better) | -0.83 (no clear difference) | +0.46 (no clear difference) | +0.18 (no clear difference) | -0.44 | 6/12 | no clear difference | 0.90 |
| S4 query pop LIF: fire_stft − fire_same ((ga): STFT bands vs the unsplit signal) | -0.66 (no clear difference) | +0.19 (no clear difference) | +0.66 (no clear difference) | -0.01 (no clear difference) | +0.05 | 5/12 | no clear difference | 0.91 |
| S4 query pop f-LIF: fire_stft − fire_same ((ga): STFT bands vs the unsplit signal) | -0.37 (no clear difference) | -0.76 (no clear difference) | +0.01 (no clear difference) | -0.08 (no clear difference) | -0.30 | 9/12 | no clear difference | 0.92 |
| report query pop LIF: stft − freq (STFT vs whole-window DCT, same structure (report only)) | +1.04 (worse) | -1.79 (better) | +0.38 (no clear difference) | +0.21 (no clear difference) | -0.04 | 4/12 | no clear difference | 0.92 |
| report query pop f-LIF: stft − freq (STFT vs whole-window DCT, same structure (report only)) | +0.26 (no clear difference) | -1.30 (better) | +0.34 (no clear difference) | +0.17 (no clear difference) | -0.13 | 6/12 | no clear difference | 0.93 |
| report query pop LIF: stft − same (STFT bands vs unsplit x4, soma structure (report only)) | -0.75 (no clear difference) | +0.30 (no clear difference) | -0.46 (no clear difference) | +0.11 (no clear difference) | -0.20 | 6/12 | no clear difference | 0.92 |
| report query pop f-LIF: stft − same (STFT bands vs unsplit x4, soma structure (report only)) | +1.61 (worse) | +0.05 (no clear difference) | -0.90 (no clear difference) | +0.03 (no clear difference) | +0.20 | 7/12 | no clear difference | 0.93 |
| report query pop LIF: fire_same − msn (soma-less tau-2 structure alone (report only)) | -0.81 (no clear difference) | +0.31 (no clear difference) | -0.62 (no clear difference) | +0.11 (no clear difference) | -0.25 | 8/12 | no clear difference | 1.09 |
| report query pop f-LIF: fire_same − msn (soma-less tau-2 structure alone (report only)) | -0.69 (no clear difference) | +0.59 (no clear difference) | -0.55 (no clear difference) | +0.16 (no clear difference) | -0.12 | 6/12 | no clear difference | 1.00 |

## MIT-BIH — test macro-F1 (%, last-epoch weights, lr 0.01)

| model | variant | seed 42 | seed 7 | seed 2026 | mean | accuracy | F1 N / S / V / F / Q |
|---|---|---|---|---|---|---|---|
| query pop LIF | msn (base) | 92.10 | 92.20 | 92.17 | 92.16 | 98.56 | 99.2 / 85.8 / 96.0 / 80.7 / 99.0 |
| query pop LIF | stft = (na): soma, tau 4-32, STFT bands | 91.86 | 91.83 | 91.76 | 91.82 | 98.47 | 99.2 / 84.3 / 96.0 / 80.8 / 98.8 |
| query pop LIF | fire_stft = (ga): no soma, tau 2, STFT bands | 92.42 | 91.92 | 92.08 | 92.14 | 98.52 | 99.2 / 84.6 / 96.1 / 81.8 / 99.0 |
| query pop LIF | fire_same: no soma, tau 2, unsplit x4 | 91.83 | 91.74 | 92.25 | 91.94 | 98.51 | 99.2 / 85.4 / 96.1 / 80.1 / 98.9 |
| query pop LIF | freq (11-13): soma, tau 4-32, DCT bands | 91.12 | 91.53 | 91.24 | 91.29 | 98.36 | 99.1 / 83.3 / 95.6 / 79.6 / 98.8 |
| query pop LIF | same (11-13): soma, tau 4-32, unsplit x4 | 92.13 | 91.92 | 92.02 | 92.02 | 98.53 | 99.2 / 85.0 / 96.1 / 80.8 / 99.0 |
| query pop f-LIF | msn (base) | 92.04 | 92.00 | 92.76 | 92.27 | 98.57 | 99.3 / 85.7 / 96.1 / 81.2 / 99.1 |
| query pop f-LIF | stft = (na): soma, tau 4-32, STFT bands | 91.43 | 91.75 | 91.28 | 91.48 | 98.43 | 99.2 / 83.5 / 95.9 / 80.1 / 98.9 |
| query pop f-LIF | fire_stft = (ga): no soma, tau 2, STFT bands | 91.73 | 91.92 | 91.26 | 91.64 | 98.47 | 99.2 / 84.7 / 96.0 / 79.5 / 98.8 |
| query pop f-LIF | fire_same: no soma, tau 2, unsplit x4 | 92.55 | 92.28 | 92.16 | 92.33 | 98.58 | 99.3 / 86.0 / 96.1 / 81.2 / 99.0 |
| query pop f-LIF | freq (11-13): soma, tau 4-32, DCT bands | 91.36 | 91.86 | 91.53 | 91.58 | 98.43 | 99.2 / 84.2 / 95.8 / 79.9 / 98.8 |
| query pop f-LIF | same (11-13): soma, tau 4-32, unsplit x4 | 91.94 | 92.24 | 91.86 | 92.01 | 98.53 | 99.2 / 85.3 / 96.2 / 80.3 / 99.0 |

## MIT-BIH — verdicts (Δ = A − B, %p)

| comparison | per seed (42 / 7 / 2026) | mean | > 0 | verdict | energy A/B |
|---|---|---|---|---|---|
| S1 query pop LIF: stft − msn ((na) vs the base) | -0.24 / -0.36 / -0.41 | -0.34 | 0/3 | no clear difference | 1.05 |
| S1 query pop f-LIF: stft − msn ((na) vs the base) | -0.61 / -0.26 / -1.48 | -0.78 | 0/3 | worse | 1.05 |
| S2 query pop LIF: fire_stft − msn ((ga) vs the base) | +0.32 / -0.28 / -0.09 | -0.02 | 1/3 | no clear difference | 0.99 |
| S2 query pop f-LIF: fire_stft − msn ((ga) vs the base) | -0.31 / -0.09 / -1.50 | -0.63 | 0/3 | worse | 0.92 |
| S3 query pop LIF: fire_stft − stft ((ga) vs (na)) | +0.55 / +0.08 / +0.32 | +0.32 | 3/3 | no clear difference | 0.94 |
| S3 query pop f-LIF: fire_stft − stft ((ga) vs (na)) | +0.30 / +0.17 / -0.02 | +0.15 | 2/3 | no clear difference | 0.88 |
| S4 query pop LIF: fire_stft − fire_same ((ga): STFT bands vs the unsplit signal) | +0.59 / +0.18 / -0.17 | +0.20 | 2/3 | no clear difference | 0.84 |
| S4 query pop f-LIF: fire_stft − fire_same ((ga): STFT bands vs the unsplit signal) | -0.82 / -0.36 / -0.90 | -0.69 | 0/3 | worse | 0.85 |
| report query pop LIF: stft − freq (STFT vs whole-window DCT, same structure (report only)) | +0.75 / +0.31 / +0.52 | +0.52 | 3/3 | better | 0.88 |
| report query pop f-LIF: stft − freq (STFT vs whole-window DCT, same structure (report only)) | +0.07 / -0.11 / -0.25 | -0.10 | 1/3 | no clear difference | 0.89 |
| report query pop LIF: stft − same (STFT bands vs unsplit x4, soma structure (report only)) | -0.27 / -0.08 / -0.26 | -0.20 | 0/3 | no clear difference | 0.90 |
| report query pop f-LIF: stft − same (STFT bands vs unsplit x4, soma structure (report only)) | -0.51 / -0.49 / -0.58 | -0.52 | 0/3 | worse | 0.91 |
| report query pop LIF: fire_same − msn (soma-less tau-2 structure alone (report only)) | -0.27 / -0.46 / +0.08 | -0.22 | 1/3 | no clear difference | 1.18 |
| report query pop f-LIF: fire_same − msn (soma-less tau-2 structure alone (report only)) | +0.51 / +0.27 / -0.60 | +0.06 | 2/3 | no clear difference | 1.09 |

## conclusions (rule of the pre-registration)

| comparison | task | pop LIF | pop f-LIF | conclusion |
|---|---|---|---|---|
| S1 stft − msn ((na) vs the base) | ETT | no clear difference | no clear difference | no clear difference |
| S1 stft − msn ((na) vs the base) | MIT-BIH | no clear difference | worse | neuron-dependent (weak) |
| S2 fire_stft − msn ((ga) vs the base) | ETT | no clear difference | no clear difference | no clear difference |
| S2 fire_stft − msn ((ga) vs the base) | MIT-BIH | no clear difference | worse | neuron-dependent (weak) |
| S3 fire_stft − stft ((ga) vs (na)) | ETT | no clear difference | no clear difference | no clear difference |
| S3 fire_stft − stft ((ga) vs (na)) | MIT-BIH | no clear difference | no clear difference | no clear difference |
| S4 fire_stft − fire_same ((ga): STFT bands vs the unsplit signal) | ETT | no clear difference | no clear difference | no clear difference |
| S4 fire_stft − fire_same ((ga): STFT bands vs the unsplit signal) | MIT-BIH | no clear difference | worse | neuron-dependent (weak) |

## report only

### firing rates, energy and uncounted costs (mean of seeds; ETT: mean over the 4 cells)

| model | variant | ETT rho_neo | ETT rho_hippo | ETT energy uJ (head recount) | ETT MS additions / transform uJ | MIT-BIH rho_neo | MIT-BIH rho_hippo | MIT-BIH energy uJ (logged) | MIT-BIH MS additions / transform uJ | params ETTh2 H96 / MIT-BIH |
|---|---|---|---|---|---|---|---|---|---|---|
| query pop LIF | msn | 0.0328 | 0.0574 | 13.04 | 0.015 / 0.000 | 0.0587 | 0.1506 | 11.37 | 0.008 / 0.000 | 175544 / 1230888 |
| query pop LIF | stft | 0.0169 | 0.0592 | 13.30 | 0.058 / 0.056 | 0.0650 | 0.1661 | 11.96 | 0.030 / 0.053 | 175928 / 1233192 |
| query pop LIF | fire_stft | 0.0604 | 0.0602 | 12.99 | 0.058 / 0.056 | 0.0336 | 0.1549 | 11.29 | 0.030 / 0.053 | 175416 / 1228584 |
| query pop LIF | fire_same | 0.0770 | 0.0592 | 14.23 | 0.058 / 0.000 | 0.0624 | 0.1478 | 13.43 | 0.030 / 0.000 | 176952 / 1265448 |
| query pop LIF | freq | 0.0186 | 0.0612 | 14.50 | 0.058 / 0.644 | 0.0594 | 0.1560 | 13.53 | 0.030 / 0.379 | 177464 / 1270056 |
| query pop LIF | same | 0.0250 | 0.0556 | 14.49 | 0.058 / 0.000 | 0.0531 | 0.1461 | 13.24 | 0.030 / 0.000 | 177464 / 1270056 |
| query pop f-LIF | msn | 0.0371 | 0.0581 | 15.86 | 0.015 / 0.000 | 0.0757 | 0.1476 | 12.57 | 0.008 / 0.000 | 175544 / 1230888 |
| query pop f-LIF | stft | 0.0208 | 0.0619 | 16.14 | 0.058 / 0.056 | 0.0880 | 0.1621 | 13.19 | 0.030 / 0.053 | 175928 / 1233192 |
| query pop f-LIF | fire_stft | 0.0567 | 0.0592 | 14.57 | 0.058 / 0.056 | 0.0286 | 0.1538 | 11.57 | 0.030 / 0.053 | 175416 / 1228584 |
| query pop f-LIF | fire_same | 0.0673 | 0.0604 | 15.80 | 0.058 / 0.000 | 0.0535 | 0.1461 | 13.67 | 0.030 / 0.000 | 176952 / 1265448 |
| query pop f-LIF | freq | 0.0233 | 0.0637 | 17.34 | 0.058 / 0.644 | 0.0886 | 0.1601 | 14.86 | 0.030 / 0.379 | 177464 / 1270056 |
| query pop f-LIF | same | 0.0283 | 0.0558 | 17.33 | 0.058 / 0.000 | 0.0803 | 0.1463 | 14.53 | 0.030 / 0.000 | 177464 / 1270056 |

### test MAE (mean of seeds)

| model | variant | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|---|
| query pop LIF | msn | 0.3356 | 0.4330 | 0.2535 | 0.3954 |
| query pop LIF | stft | 0.3360 | 0.4373 | 0.2533 | 0.3949 |
| query pop LIF | fire_stft | 0.3344 | 0.4350 | 0.2537 | 0.3958 |
| query pop LIF | fire_same | 0.3347 | 0.4346 | 0.2531 | 0.3959 |
| query pop LIF | freq | 0.3341 | 0.4414 | 0.2533 | 0.3950 |
| query pop LIF | same | 0.3357 | 0.4344 | 0.2534 | 0.3954 |
| query pop f-LIF | msn | 0.3347 | 0.4338 | 0.2540 | 0.3956 |
| query pop f-LIF | stft | 0.3360 | 0.4382 | 0.2533 | 0.3949 |
| query pop f-LIF | fire_stft | 0.3331 | 0.4335 | 0.2537 | 0.3957 |
| query pop f-LIF | fire_same | 0.3338 | 0.4351 | 0.2536 | 0.3966 |
| query pop f-LIF | freq | 0.3355 | 0.4404 | 0.2530 | 0.3950 |
| query pop f-LIF | same | 0.3331 | 0.4357 | 0.2538 | 0.3957 |

### every model vs Spikformer-style (flif-v1-gen-ett / flif-v1-gen-mitbih; same rules)

| model | ETT: ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | ETT R % | lower | ETT verdict | ETT energy ratio | MIT-BIH Δ %p | MIT-BIH verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| query pop LIF msn | +0.96 | -1.57 | +0.08 | +0.04 | -0.12 | 6/12 | no clear difference | 1.25 | -0.07 | no clear difference |
| query pop LIF stft | +0.51 | -0.71 | -0.41 | -0.09 | -0.18 | 7/12 | no clear difference | 1.27 | -0.40 | no clear difference |
| query pop LIF fire_stft | -0.55 | -1.08 | +0.11 | +0.14 | -0.35 | 6/12 | no clear difference | 1.24 | -0.08 | no clear difference |
| query pop LIF fire_same | +0.11 | -1.27 | -0.55 | +0.15 | -0.39 | 8/12 | no clear difference | 1.36 | -0.28 | no clear difference |
| query pop LIF freq | -0.53 | +1.14 | -0.78 | -0.30 | -0.12 | 9/12 | no clear difference | 1.39 | -0.93 | worse |
| query pop LIF same | +1.29 | -1.01 | +0.06 | -0.20 | +0.03 | 7/12 | no clear difference | 1.39 | -0.20 | no clear difference |
| query pop f-LIF msn | +0.04 | -1.35 | +0.43 | +0.03 | -0.21 | 6/12 | no clear difference | 1.51 | +0.04 | no clear difference |
| query pop f-LIF stft | +0.57 | -0.69 | -0.57 | -0.07 | -0.19 | 6/12 | no clear difference | 1.54 | -0.74 | worse |
| query pop f-LIF fire_stft | -1.04 | -1.54 | -0.12 | +0.11 | -0.65 | 9/12 | no clear difference | 1.39 | -0.58 | worse |
| query pop f-LIF fire_same | -0.66 | -0.77 | -0.13 | +0.19 | -0.34 | 5/12 | no clear difference | 1.51 | +0.11 | no clear difference |
| query pop f-LIF freq | +0.31 | +0.64 | -0.91 | -0.23 | -0.05 | 8/12 | no clear difference | 1.66 | -0.64 | worse |
| query pop f-LIF same | -1.00 | -0.74 | +0.34 | -0.10 | -0.37 | 8/12 | no clear difference | 1.66 | -0.21 | no clear difference |
