# Frequency-band test (Neocortex-MS query population LIF / f-LIF) — summary

See the module docstring and docs/MODI_PROJECT_LOG.md 2026-11-13 for the pre-registered rules.

## ETT — test MSE (mean ± sd of seeds 42 / 7 / 2026, best-validation weights)

| model | variant | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean validation L1 |
|---|---|---|---|---|---|---|
| query pop LIF | msn (base: Neocortex MS, one shared current) | 0.2931 ± 0.0045 | 0.4157 ± 0.0011 | 0.1776 ± 0.0004 | 0.4036 ± 0.0005 | 0.3624 |
| query pop LIF | freq (bands, lowest -> tau 32) | 0.2888 ± 0.0023 | 0.4272 ± 0.0154 | 0.1761 ± 0.0005 | 0.4022 ± 0.0019 | 0.3602 |
| query pop LIF | same (unsplit signal, 4 embeddings) | 0.2941 ± 0.0041 | 0.4181 ± 0.0026 | 0.1776 ± 0.0016 | 0.4026 ± 0.0024 | 0.3619 |
| query pop LIF | rev (bands, lowest -> tau 4) | 0.2903 ± 0.0022 | 0.4173 ± 0.0044 | 0.1769 ± 0.0002 | 0.4034 ± 0.0009 | 0.3601 |
| query pop f-LIF | msn (base: Neocortex MS, one shared current) | 0.2904 ± 0.0039 | 0.4166 ± 0.0055 | 0.1782 ± 0.0011 | 0.4036 ± 0.0031 | 0.3618 |
| query pop f-LIF | freq (bands, lowest -> tau 32) | 0.2912 ± 0.0036 | 0.4251 ± 0.0144 | 0.1758 ± 0.0009 | 0.4025 ± 0.0016 | 0.3603 |
| query pop f-LIF | same (unsplit signal, 4 embeddings) | 0.2874 ± 0.0065 | 0.4192 ± 0.0062 | 0.1781 ± 0.0011 | 0.4031 ± 0.0027 | 0.3614 |
| query pop f-LIF | rev (bands, lowest -> tau 4) | 0.2895 ± 0.0018 | 0.4151 ± 0.0028 | 0.1763 ± 0.0009 | 0.4025 ± 0.0015 | 0.3605 |

## ETT — verdicts (r = A / B − 1 of test MSE, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B |
|---|---|---|---|---|---|---|---|---|
| F1 query pop LIF: freq − msn (the proposal vs the base) | -1.47 (better) | +2.76 (worse) | -0.86 (no clear difference) | -0.34 (no clear difference) | +0.02 | 9/12 | no clear difference | 1.11 |
| F1 query pop f-LIF: freq − msn (the proposal vs the base) | +0.28 (no clear difference) | +2.05 (worse) | -1.33 (better) | -0.26 (no clear difference) | +0.18 | 8/12 | no clear difference | 1.10 |
| F2 query pop LIF: freq − same (band split vs 4 embeddings of the unsplit signal) | -1.78 (better) | +2.17 (worse) | -0.84 (no clear difference) | -0.10 (no clear difference) | -0.14 | 9/12 | no clear difference | 1.00 |
| F2 query pop f-LIF: freq − same (band split vs 4 embeddings of the unsplit signal) | +1.34 (worse) | +1.38 (worse) | -1.24 (better) | -0.13 (no clear difference) | +0.34 | 7/12 | no clear difference | 1.00 |
| F3 query pop LIF: freq − rev (lowest band on the slowest vs on the fastest member) | -0.52 (no clear difference) | +2.36 (worse) | -0.46 (no clear difference) | -0.29 (no clear difference) | +0.27 | 8/12 | no clear difference | 1.01 |
| F3 query pop f-LIF: freq − rev (lowest band on the slowest vs on the fastest member) | +0.59 (no clear difference) | +2.40 (worse) | -0.28 (no clear difference) | -0.00 (no clear difference) | +0.68 | 6/12 | no clear difference | 1.01 |
| report query pop LIF: same − msn (4 embeddings alone (report only)) | +0.32 (no clear difference) | +0.58 (no clear difference) | -0.02 (no clear difference) | -0.24 (no clear difference) | +0.16 | 4/12 | no clear difference | 1.11 |
| report query pop f-LIF: same − msn (4 embeddings alone (report only)) | -1.03 (better) | +0.64 (no clear difference) | -0.09 (no clear difference) | -0.13 (no clear difference) | -0.15 | 8/12 | no clear difference | 1.09 |
| report query pop LIF: rev − msn (reversed pairing vs the base (report only)) | -0.94 (no clear difference) | +0.38 (no clear difference) | -0.40 (no clear difference) | -0.05 (no clear difference) | -0.25 | 7/12 | no clear difference | 1.11 |
| report query pop f-LIF: rev − msn (reversed pairing vs the base (report only)) | -0.31 (no clear difference) | -0.36 (no clear difference) | -1.06 (better) | -0.26 (no clear difference) | -0.50 | 8/12 | no clear difference | 1.08 |

## MIT-BIH — test macro-F1 (%, last-epoch weights, lr 0.01)

| model | variant | seed 42 | seed 7 | seed 2026 | mean | accuracy | F1 N / S / V / F / Q |
|---|---|---|---|---|---|---|---|
| query pop LIF | msn (base: Neocortex MS, one shared current) | 92.10 | 92.20 | 92.17 | 92.16 | 98.56 | 99.2 / 85.8 / 96.0 / 80.7 / 99.0 |
| query pop LIF | freq (bands, lowest -> tau 32) | 91.12 | 91.53 | 91.24 | 91.29 | 98.36 | 99.1 / 83.3 / 95.6 / 79.6 / 98.8 |
| query pop LIF | same (unsplit signal, 4 embeddings) | 92.13 | 91.92 | 92.02 | 92.02 | 98.53 | 99.2 / 85.0 / 96.1 / 80.8 / 99.0 |
| query pop LIF | rev (bands, lowest -> tau 4) | 91.50 | 91.95 | 91.29 | 91.58 | 98.44 | 99.2 / 84.0 / 96.0 / 79.9 / 98.8 |
| query pop f-LIF | msn (base: Neocortex MS, one shared current) | 92.04 | 92.00 | 92.76 | 92.27 | 98.57 | 99.3 / 85.7 / 96.1 / 81.2 / 99.1 |
| query pop f-LIF | freq (bands, lowest -> tau 32) | 91.36 | 91.86 | 91.53 | 91.58 | 98.43 | 99.2 / 84.2 / 95.8 / 79.9 / 98.8 |
| query pop f-LIF | same (unsplit signal, 4 embeddings) | 91.94 | 92.24 | 91.86 | 92.01 | 98.53 | 99.2 / 85.3 / 96.2 / 80.3 / 99.0 |
| query pop f-LIF | rev (bands, lowest -> tau 4) | 91.76 | 92.02 | 91.69 | 91.82 | 98.48 | 99.2 / 84.4 / 96.0 / 80.6 / 98.9 |

## MIT-BIH — verdicts (Δ = A − B, %p)

| comparison | per seed (42 / 7 / 2026) | mean | > 0 | verdict | energy A/B |
|---|---|---|---|---|---|
| F1 query pop LIF: freq − msn (the proposal vs the base) | -0.98 / -0.67 / -0.93 | -0.86 | 0/3 | worse | 1.19 |
| F1 query pop f-LIF: freq − msn (the proposal vs the base) | -0.68 / -0.15 / -1.23 | -0.68 | 0/3 | worse | 1.18 |
| F2 query pop LIF: freq − same (band split vs 4 embeddings of the unsplit signal) | -1.01 / -0.39 / -0.78 | -0.73 | 0/3 | worse | 1.02 |
| F2 query pop f-LIF: freq − same (band split vs 4 embeddings of the unsplit signal) | -0.58 / -0.38 / -0.33 | -0.43 | 0/3 | no clear difference | 1.02 |
| F3 query pop LIF: freq − rev (lowest band on the slowest vs on the fastest member) | -0.39 / -0.42 / -0.05 | -0.29 | 0/3 | no clear difference | 1.00 |
| F3 query pop f-LIF: freq − rev (lowest band on the slowest vs on the fastest member) | -0.41 / -0.16 / -0.16 | -0.24 | 0/3 | no clear difference | 1.02 |
| report query pop LIF: same − msn (4 embeddings alone (report only)) | +0.03 / -0.28 / -0.15 | -0.13 | 1/3 | no clear difference | 1.16 |
| report query pop f-LIF: same − msn (4 embeddings alone (report only)) | -0.10 / +0.23 / -0.90 | -0.26 | 1/3 | no clear difference | 1.16 |
| report query pop LIF: rev − msn (reversed pairing vs the base (report only)) | -0.60 / -0.25 / -0.88 | -0.57 | 0/3 | worse | 1.19 |
| report query pop f-LIF: rev − msn (reversed pairing vs the base (report only)) | -0.27 / +0.01 / -1.07 | -0.44 | 1/3 | no clear difference | 1.16 |

## conclusions (rule of the pre-registration)

| comparison | task | pop LIF | pop f-LIF | conclusion |
|---|---|---|---|---|
| F1 freq − msn (the proposal vs the base) | ETT | no clear difference | no clear difference | no clear difference |
| F1 freq − msn (the proposal vs the base) | MIT-BIH | worse | worse | worse |
| F2 freq − same (band split vs 4 embeddings of the unsplit signal) | ETT | no clear difference | no clear difference | no clear difference |
| F2 freq − same (band split vs 4 embeddings of the unsplit signal) | MIT-BIH | worse | no clear difference | neuron-dependent (weak) |
| F3 freq − rev (lowest band on the slowest vs on the fastest member) | ETT | no clear difference | no clear difference | no clear difference |
| F3 freq − rev (lowest band on the slowest vs on the fastest member) | MIT-BIH | no clear difference | no clear difference | no clear difference |

## report only

### firing rates, energy and uncounted costs (mean of seeds; ETT: mean over the 4 cells)

| model | variant | ETT rho_neo | ETT rho_hippo | ETT energy uJ (head recount) | ETT MS additions / DCT uJ | MIT-BIH rho_neo | MIT-BIH rho_hippo | MIT-BIH energy uJ (logged) | MIT-BIH MS additions / DCT uJ | params ETTh2 H96 / MIT-BIH |
|---|---|---|---|---|---|---|---|---|---|---|
| query pop LIF | msn | 0.0328 | 0.0574 | 13.04 | 0.015 / 0.000 | 0.0587 | 0.1506 | 11.37 | 0.008 / 0.000 | 175544 / 1230888 |
| query pop LIF | freq | 0.0186 | 0.0612 | 14.50 | 0.058 / 0.644 | 0.0594 | 0.1560 | 13.53 | 0.030 / 0.379 | 177464 / 1270056 |
| query pop LIF | same | 0.0250 | 0.0556 | 14.49 | 0.058 / 0.000 | 0.0531 | 0.1461 | 13.24 | 0.030 / 0.000 | 177464 / 1270056 |
| query pop LIF | rev | 0.0133 | 0.0562 | 14.44 | 0.058 / 0.644 | 0.0541 | 0.1517 | 13.49 | 0.030 / 0.379 | 177464 / 1270056 |
| query pop f-LIF | msn | 0.0371 | 0.0581 | 15.86 | 0.015 / 0.000 | 0.0757 | 0.1476 | 12.57 | 0.008 / 0.000 | 175544 / 1230888 |
| query pop f-LIF | freq | 0.0233 | 0.0637 | 17.34 | 0.058 / 0.644 | 0.0886 | 0.1601 | 14.86 | 0.030 / 0.379 | 177464 / 1270056 |
| query pop f-LIF | same | 0.0283 | 0.0558 | 17.33 | 0.058 / 0.000 | 0.0803 | 0.1463 | 14.53 | 0.030 / 0.000 | 177464 / 1270056 |
| query pop f-LIF | rev | 0.0020 | 0.0519 | 17.16 | 0.058 / 0.644 | 0.0701 | 0.1488 | 14.60 | 0.030 / 0.379 | 177464 / 1270056 |

### test MAE (mean of seeds)

| model | variant | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|---|
| query pop LIF | msn | 0.3356 | 0.4330 | 0.2535 | 0.3954 |
| query pop LIF | freq | 0.3341 | 0.4414 | 0.2533 | 0.3950 |
| query pop LIF | same | 0.3357 | 0.4344 | 0.2534 | 0.3954 |
| query pop LIF | rev | 0.3350 | 0.4364 | 0.2532 | 0.3957 |
| query pop f-LIF | msn | 0.3347 | 0.4338 | 0.2540 | 0.3956 |
| query pop f-LIF | freq | 0.3355 | 0.4404 | 0.2530 | 0.3950 |
| query pop f-LIF | same | 0.3331 | 0.4357 | 0.2538 | 0.3957 |
| query pop f-LIF | rev | 0.3344 | 0.4350 | 0.2528 | 0.3950 |

### every model vs Spikformer-style (flif-v1-gen-ett / flif-v1-gen-mitbih; same rules)

| model | ETT: ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | ETT R % | lower | ETT verdict | ETT energy ratio | MIT-BIH Δ %p | MIT-BIH verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| query pop LIF msn | +0.96 | -1.57 | +0.08 | +0.04 | -0.12 | 6/12 | no clear difference | 1.25 | -0.07 | no clear difference |
| query pop LIF freq | -0.53 | +1.14 | -0.78 | -0.30 | -0.12 | 9/12 | no clear difference | 1.39 | -0.93 | worse |
| query pop LIF same | +1.29 | -1.01 | +0.06 | -0.20 | +0.03 | 7/12 | no clear difference | 1.39 | -0.20 | no clear difference |
| query pop LIF rev | -0.01 | -1.19 | -0.33 | -0.01 | -0.39 | 10/12 | no clear difference | 1.38 | -0.64 | worse |
| query pop f-LIF msn | +0.04 | -1.35 | +0.43 | +0.03 | -0.21 | 6/12 | no clear difference | 1.51 | +0.04 | no clear difference |
| query pop f-LIF freq | +0.31 | +0.64 | -0.91 | -0.23 | -0.05 | 8/12 | no clear difference | 1.66 | -0.64 | worse |
| query pop f-LIF same | -1.00 | -0.74 | +0.34 | -0.10 | -0.37 | 8/12 | no clear difference | 1.66 | -0.21 | no clear difference |
| query pop f-LIF rev | -0.29 | -1.72 | -0.63 | -0.23 | -0.72 | 10/12 | no clear difference | 1.64 | -0.40 | no clear difference |
