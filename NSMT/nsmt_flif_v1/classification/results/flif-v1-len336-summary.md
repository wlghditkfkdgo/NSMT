# Input-length test (Neocortex-MS query population LIF / f-LIF vs Spikformer-style; input 96 / 336) — summary

See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. Tokens: L 96 = 24, L 336 = 84.

## ETT — test MSE (mean ± sd of seeds 42 / 7 / 2026, best-validation weights)

| model | input | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean validation L1 |
|---|---|---|---|---|---|---|
| Spikformer-style | 96 | 0.2903 ± 0.0012 | 0.4223 ± 0.0024 | 0.1775 ± 0.0007 | 0.4034 ± 0.0016 | 0.3612 |
| Spikformer-style | 336 | 0.3094 ± 0.0127 | 0.3983 ± 0.0018 | 0.1744 ± 0.0013 | 0.3672 ± 0.0018 | 0.3649 |
| query pop LIF base (msn) | 96 | 0.2931 ± 0.0045 | 0.4157 ± 0.0011 | 0.1776 ± 0.0004 | 0.4036 ± 0.0005 | 0.3624 |
| query pop LIF base (msn) | 336 | 0.3086 ± 0.0064 | 0.3987 ± 0.0027 | 0.1714 ± 0.0011 | 0.3633 ± 0.0003 | 0.3651 |
| query pop LIF nomem (no state across tokens) | 96 | 0.2902 ± 0.0042 | 0.4174 ± 0.0066 | 0.1768 ± 0.0004 | 0.4034 ± 0.0026 | 0.3606 |
| query pop LIF nomem (no state across tokens) | 336 | 0.2975 ± 0.0080 | 0.3948 ± 0.0011 | 0.1765 ± 0.0036 | 0.3661 ± 0.0017 | 0.3646 |
| query pop LIF nshuf (shuffled token order) | 96 | 0.2921 ± 0.0033 | 0.4123 ± 0.0026 | 0.1766 ± 0.0011 | 0.4029 ± 0.0019 | 0.3611 |
| query pop LIF nshuf (shuffled token order) | 336 | 0.2925 ± 0.0062 | 0.3934 ± 0.0009 | 0.1760 ± 0.0054 | 0.3723 ± 0.0027 | 0.3645 |
| query pop f-LIF base (msn) | 96 | 0.2904 ± 0.0039 | 0.4166 ± 0.0055 | 0.1782 ± 0.0011 | 0.4036 ± 0.0031 | 0.3618 |
| query pop f-LIF base (msn) | 336 | 0.2952 ± 0.0074 | 0.3967 ± 0.0029 | 0.1708 ± 0.0049 | 0.3694 ± 0.0019 | 0.3660 |
| query pop f-LIF nomem (no state across tokens) | 96 | 0.2906 ± 0.0031 | 0.4156 ± 0.0045 | 0.1766 ± 0.0015 | 0.4024 ± 0.0025 | 0.3607 |
| query pop f-LIF nomem (no state across tokens) | 336 | 0.2920 ± 0.0083 | 0.4104 ± 0.0080 | 0.1749 ± 0.0053 | 0.3661 ± 0.0049 | 0.3640 |
| query pop f-LIF nshuf (shuffled token order) | 96 | 0.2910 ± 0.0040 | 0.4160 ± 0.0036 | 0.1764 ± 0.0002 | 0.4036 ± 0.0022 | 0.3605 |
| query pop f-LIF nshuf (shuffled token order) | 336 | 0.2954 ± 0.0094 | 0.3911 ± 0.0014 | 0.1740 ± 0.0052 | 0.3649 ± 0.0045 | 0.3647 |

## L1 — base − Spikformer-style at each input length (r = A / B − 1 of test MSE, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B |
|---|---|---|---|---|---|---|---|---|
| L1 query pop LIF − Spikformer-style, input 96 | +0.96 (no clear difference) | -1.57 (better) | +0.08 (no clear difference) | +0.04 (no clear difference) | -0.12 | 6/12 | no clear difference | 1.25 |
| L1 query pop LIF − Spikformer-style, input 336 | -0.18 (no clear difference) | +0.11 (no clear difference) | -1.68 (better) | -1.07 (better) | -0.71 | 10/12 | no clear difference | 1.27 |
| L1 query pop f-LIF − Spikformer-style, input 96 | +0.04 (no clear difference) | -1.35 (better) | +0.43 (no clear difference) | +0.03 (no clear difference) | -0.21 | 6/12 | no clear difference | 1.51 |
| L1 query pop f-LIF − Spikformer-style, input 336 | -4.48 (better) | -0.38 (no clear difference) | -2.03 (better) | +0.60 (no clear difference) | -1.57 | 8/12 | better | 1.65 |

## L2 — change of the gap with the input length (g = r_L − r_shorter per cell and seed, %p; negative = the base gains more)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean g | pairs negative | verdict |
|---|---|---|---|---|---|---|---|
| L2 query pop LIF: gap at 336 − gap at 96 | -1.15 | +1.68 | -1.76 | -1.11 | -0.58 | 8/12 | no clear change |
| L2 query pop f-LIF: gap at 336 − gap at 96 | -4.52 | +0.97 | -2.46 | +0.57 | -1.36 | 5/12 | no clear change |

## L3 — controls of the base model at each input length (r = A / B − 1 of test MSE, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B |
|---|---|---|---|---|---|---|---|---|
| L3 query pop LIF: msn − nshuf (order), input 96 | +0.35 (no clear difference) | +0.83 (no clear difference) | +0.57 (no clear difference) | +0.18 (no clear difference) | +0.48 | 4/12 | no clear difference | 1.00 |
| L3 query pop LIF: msn − nshuf (order), input 336 | +5.56 (worse) | +1.34 (worse) | -2.50 (better) | -2.42 (better) | +0.50 | 5/12 | no clear difference | 1.01 |
| L3 query pop f-LIF: msn − nshuf (order), input 96 | -0.19 (no clear difference) | +0.14 (no clear difference) | +1.02 (worse) | +0.00 (no clear difference) | +0.24 | 4/12 | no clear difference | 1.00 |
| L3 query pop f-LIF: msn − nshuf (order), input 336 | -0.03 (no clear difference) | +1.45 (worse) | -1.71 (better) | +1.25 (worse) | +0.24 | 3/12 | no clear difference | 1.00 |
| L3 query pop LIF: msn − nomem (memory), input 96 | +1.02 (worse) | -0.41 (no clear difference) | +0.45 (no clear difference) | +0.05 (no clear difference) | +0.28 | 4/12 | no clear difference | 1.04 |
| L3 query pop LIF: msn − nomem (memory), input 336 | +3.80 (worse) | +1.00 (worse) | -2.86 (better) | -0.76 (no clear difference) | +0.30 | 5/12 | no clear difference | 1.06 |
| L3 query pop f-LIF: msn − nomem (memory), input 96 | -0.06 (no clear difference) | +0.24 (no clear difference) | +0.93 (no clear difference) | +0.29 (no clear difference) | +0.35 | 4/12 | no clear difference | 1.22 |
| L3 query pop f-LIF: msn − nomem (memory), input 336 | +1.09 (worse) | -3.31 (better) | -2.30 (better) | +0.91 (no clear difference) | -0.90 | 7/12 | no clear difference | 1.31 |
| report query pop LIF: nshuf − nomem (report only), input 96 | +0.67 (no clear difference) | -1.22 (better) | -0.11 (no clear difference) | -0.13 (no clear difference) | -0.20 | 6/12 | no clear difference | 1.04 |
| report query pop LIF: nshuf − nomem (report only), input 336 | -1.64 (better) | -0.34 (no clear difference) | -0.34 (no clear difference) | +1.71 (worse) | -0.15 | 7/12 | no clear difference | 1.05 |
| report query pop f-LIF: nshuf − nomem (report only), input 96 | +0.13 (no clear difference) | +0.10 (no clear difference) | -0.09 (no clear difference) | +0.29 (no clear difference) | +0.11 | 3/12 | no clear difference | 1.22 |
| report query pop f-LIF: nshuf − nomem (report only), input 336 | +1.16 (no clear difference) | -4.69 (better) | -0.42 (no clear difference) | -0.33 (no clear difference) | -1.07 | 9/12 | better | 1.31 |

## L4 (report only) — every model against itself at the shorter input (same test targets; r = A / B − 1, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B |
|---|---|---|---|---|---|---|---|---|
| Spikformer-style: input 336 − input 96 | +6.59 (worse) | -5.69 (better) | -1.74 (better) | -8.98 (better) | -2.45 | 9/12 | better | 3.36 |
| query pop LIF base (msn): input 336 − input 96 | +5.29 (worse) | -4.08 (better) | -3.47 (better) | -9.99 (better) | -3.06 | 9/12 | better | 3.43 |
| query pop LIF nomem (no state across tokens): input 336 − input 96 | +2.55 (worse) | -5.42 (better) | -0.15 (no clear difference) | -9.25 (better) | -3.07 | 8/12 | better | 3.36 |
| query pop LIF nshuf (shuffled token order): input 336 − input 96 | +0.15 (no clear difference) | -4.57 (better) | -0.37 (no clear difference) | -7.59 (better) | -3.10 | 9/12 | better | 3.40 |
| query pop f-LIF base (msn): input 336 − input 96 | +1.66 (worse) | -4.76 (better) | -4.13 (better) | -8.46 (better) | -3.92 | 10/12 | better | 3.66 |
| query pop f-LIF nomem (no state across tokens): input 336 − input 96 | +0.52 (no clear difference) | -1.24 (better) | -0.99 (no clear difference) | -9.02 (better) | -2.68 | 8/12 | better | 3.39 |
| query pop f-LIF nshuf (shuffled token order): input 336 − input 96 | +1.54 (worse) | -6.00 (better) | -1.40 (better) | -9.58 (better) | -3.86 | 9/12 | better | 3.64 |

## conclusions (rule of the pre-registration)

| comparison | pop LIF | pop f-LIF | conclusion |
|---|---|---|---|
| L1 base − Spikformer-style, input 336 | no clear difference | better | neuron-dependent (weak) |
| L2 gap at 336 − gap at 96 | no clear change | no clear change | no clear change |
| L3 msn − nshuf (order), input 96 | no clear difference | no clear difference | no clear difference |
| L3 msn − nshuf (order), input 336 | no clear difference | no clear difference | no clear difference |
| L3 msn − nomem (memory), input 96 | no clear difference | no clear difference | no clear difference |
| L3 msn − nomem (memory), input 336 | no clear difference | no clear difference | no clear difference |

## report only

### firing rates, energy, parameters, run time (mean of seeds)

| model | input | rho_neo (4 cells) | rho_neo mean | rho_hippo mean | energy uJ (head recount, mean of 4 cells) | energy / Spikformer-style (mean of per-cell ratios) | params ETTh2 H96 / ETTm2 H720 | run time s (min–max) |
|---|---|---|---|---|---|---|---|---|
| Spikformer-style | 96 | — | — | 0.0560 | 10.46 | 1.00 | 169656 / 1132656 | 234–859 |
| Spikformer-style | 336 | — | — | 0.0534 | 35.07 | 1.00 | 539256 / 3899376 | 161–805 |
| query pop LIF base (msn) | 96 | 0.0182 / 0.0170 / 0.0537 / 0.0422 | 0.0328 | 0.0574 | 13.04 | 1.25 | 175544 / 1143024 | 172–1179 |
| query pop LIF base (msn) | 336 | 0.0664 / 0.0573 / 0.0889 / 0.0986 | 0.0778 | 0.0550 | 44.70 | 1.27 | 545144 / 3909744 | 580–7066 |
| query pop LIF nomem (no state across tokens) | 96 | 0.0000 / 0.0000 / 0.0002 / 0.0001 | 0.0001 | 0.0511 | 12.60 | 1.20 | 175544 / 1143024 | 109–529 |
| query pop LIF nomem (no state across tokens) | 336 | 0.0000 / 0.0001 / 0.0002 / 0.0002 | 0.0001 | 0.0475 | 42.27 | 1.20 | 545144 / 3909744 | 150–1061 |
| query pop LIF nshuf (shuffled token order) | 96 | 0.0134 / 0.0218 / 0.0526 / 0.0222 | 0.0275 | 0.0608 | 13.06 | 1.25 | 175544 / 1143024 | 195–999 |
| query pop LIF nshuf (shuffled token order) | 336 | 0.0322 / 0.0221 / 0.0443 / 0.0558 | 0.0386 | 0.0542 | 44.34 | 1.26 | 545144 / 3909744 | 373–6666 |
| query pop f-LIF base (msn) | 96 | 0.0220 / 0.0194 / 0.0576 / 0.0496 | 0.0371 | 0.0581 | 15.86 | 1.51 | 175544 / 1143024 | 177–1039 |
| query pop f-LIF base (msn) | 336 | 0.0552 / 0.0496 / 0.0785 / 0.0783 | 0.0654 | 0.0529 | 57.96 | 1.65 | 545144 / 3909744 | 721–6611 |
| query pop f-LIF nomem (no state across tokens) | 96 | 0.0036 / 0.0034 / 0.0028 / 0.0033 | 0.0033 | 0.0540 | 12.99 | 1.24 | 175544 / 1143024 | 135–451 |
| query pop f-LIF nomem (no state across tokens) | 336 | 0.0024 / 0.0035 / 0.0037 / 0.0045 | 0.0035 | 0.0492 | 43.99 | 1.25 | 545144 / 3909744 | 175–1382 |
| query pop f-LIF nshuf (shuffled token order) | 96 | 0.0131 / 0.0229 / 0.0627 / 0.0244 | 0.0308 | 0.0592 | 15.87 | 1.52 | 175544 / 1143024 | 176–1283 |
| query pop f-LIF nshuf (shuffled token order) | 336 | 0.0282 / 0.0240 / 0.0430 / 0.0424 | 0.0344 | 0.0535 | 57.72 | 1.64 | 545144 / 3909744 | 379–6742 |

### energy by cell (uJ, head recount, mean of seeds)

| model | input | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|---|
| Spikformer-style | 96 | 9.30 | 10.79 | 10.28 | 11.48 |
| Spikformer-style | 336 | 32.26 | 35.02 | 35.63 | 37.37 |
| query pop LIF base (msn) | 96 | 11.30 | 13.00 | 13.35 | 14.53 |
| query pop LIF base (msn) | 336 | 39.96 | 43.29 | 46.41 | 49.16 |
| query pop LIF nomem (no state across tokens) | 96 | 10.86 | 12.63 | 12.92 | 14.01 |
| query pop LIF nomem (no state across tokens) | 336 | 37.83 | 40.27 | 44.61 | 46.37 |
| query pop LIF nshuf (shuffled token order) | 96 | 11.32 | 12.98 | 13.45 | 14.48 |
| query pop LIF nshuf (shuffled token order) | 336 | 39.65 | 42.56 | 46.31 | 48.82 |
| query pop f-LIF base (msn) | 96 | 13.56 | 15.16 | 16.77 | 17.94 |
| query pop f-LIF base (msn) | 336 | 50.50 | 53.69 | 62.48 | 65.15 |
| query pop f-LIF nomem (no state across tokens) | 96 | 11.33 | 12.96 | 13.26 | 14.39 |
| query pop f-LIF nomem (no state across tokens) | 336 | 39.26 | 43.29 | 45.64 | 47.79 |
| query pop f-LIF nshuf (shuffled token order) | 96 | 13.55 | 15.22 | 16.86 | 17.83 |
| query pop f-LIF nshuf (shuffled token order) | 336 | 50.30 | 53.40 | 62.53 | 64.66 |

### test MAE (mean of seeds)

| model | input | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|---|
| Spikformer-style | 96 | 0.3350 | 0.4366 | 0.2536 | 0.3959 |
| Spikformer-style | 336 | 0.3534 | 0.4239 | 0.2551 | 0.3844 |
| query pop LIF base (msn) | 96 | 0.3356 | 0.4330 | 0.2535 | 0.3954 |
| query pop LIF base (msn) | 336 | 0.3519 | 0.4253 | 0.2545 | 0.3819 |
| query pop LIF nomem (no state across tokens) | 96 | 0.3347 | 0.4355 | 0.2530 | 0.3951 |
| query pop LIF nomem (no state across tokens) | 336 | 0.3470 | 0.4232 | 0.2566 | 0.3831 |
| query pop LIF nshuf (shuffled token order) | 96 | 0.3361 | 0.4323 | 0.2530 | 0.3954 |
| query pop LIF nshuf (shuffled token order) | 336 | 0.3435 | 0.4225 | 0.2569 | 0.3859 |
| query pop f-LIF base (msn) | 96 | 0.3347 | 0.4338 | 0.2540 | 0.3956 |
| query pop f-LIF base (msn) | 336 | 0.3465 | 0.4236 | 0.2540 | 0.3855 |
| query pop f-LIF nomem (no state across tokens) | 96 | 0.3350 | 0.4342 | 0.2530 | 0.3948 |
| query pop f-LIF nomem (no state across tokens) | 336 | 0.3447 | 0.4319 | 0.2562 | 0.3830 |
| query pop f-LIF nshuf (shuffled token order) | 96 | 0.3355 | 0.4351 | 0.2525 | 0.3953 |
| query pop f-LIF nshuf (shuffled token order) | 336 | 0.3457 | 0.4212 | 0.2550 | 0.3817 |
