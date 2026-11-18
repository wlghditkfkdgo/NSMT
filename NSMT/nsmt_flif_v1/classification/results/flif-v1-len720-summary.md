# Input-length test (Neocortex-MS query population LIF / f-LIF vs Spikformer-style; input 96 / 336 / 720) — summary

See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. Tokens: L 96 = 24, L 336 = 84, L 720 = 180.

## ETT — test MSE (mean ± sd of seeds 42 / 7 / 2026, best-validation weights)

| model | input | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean validation L1 |
|---|---|---|---|---|---|---|
| Spikformer-style | 96 | 0.2903 ± 0.0012 | 0.4223 ± 0.0024 | 0.1775 ± 0.0007 | 0.4034 ± 0.0016 | 0.3612 |
| Spikformer-style | 336 | 0.3094 ± 0.0127 | 0.3983 ± 0.0018 | 0.1744 ± 0.0013 | 0.3672 ± 0.0018 | 0.3649 |
| Spikformer-style | 720 | 0.2991 ± 0.0044 | 0.4072 ± 0.0105 | 0.1757 ± 0.0026 | 0.3896 ± 0.0073 | 0.3785 |
| query pop LIF base (msn) | 96 | 0.2931 ± 0.0045 | 0.4157 ± 0.0011 | 0.1776 ± 0.0004 | 0.4036 ± 0.0005 | 0.3624 |
| query pop LIF base (msn) | 336 | 0.3086 ± 0.0064 | 0.3987 ± 0.0027 | 0.1714 ± 0.0011 | 0.3633 ± 0.0003 | 0.3651 |
| query pop LIF base (msn) | 720 | 0.2884 ± 0.0036 | 0.4427 ± 0.0209 | 0.1771 ± 0.0017 | 0.3740 ± 0.0057 | 0.3806 |
| query pop LIF nomem (no state across tokens) | 96 | 0.2902 ± 0.0042 | 0.4174 ± 0.0066 | 0.1768 ± 0.0004 | 0.4034 ± 0.0026 | 0.3606 |
| query pop LIF nomem (no state across tokens) | 336 | 0.2975 ± 0.0080 | 0.3948 ± 0.0011 | 0.1765 ± 0.0036 | 0.3661 ± 0.0017 | 0.3646 |
| query pop LIF nomem (no state across tokens) | 720 | 0.2966 ± 0.0020 | 0.4064 ± 0.0048 | 0.1791 ± 0.0021 | 0.3805 ± 0.0045 | 0.3766 |
| query pop LIF nshuf (shuffled token order) | 96 | 0.2921 ± 0.0033 | 0.4123 ± 0.0026 | 0.1766 ± 0.0011 | 0.4029 ± 0.0019 | 0.3611 |
| query pop LIF nshuf (shuffled token order) | 336 | 0.2925 ± 0.0062 | 0.3934 ± 0.0009 | 0.1760 ± 0.0054 | 0.3723 ± 0.0027 | 0.3645 |
| query pop LIF nshuf (shuffled token order) | 720 | 0.2920 ± 0.0015 | 0.4104 ± 0.0075 | 0.1792 ± 0.0036 | 0.3787 ± 0.0131 | 0.3775 |
| query pop f-LIF base (msn) | 96 | 0.2904 ± 0.0039 | 0.4166 ± 0.0055 | 0.1782 ± 0.0011 | 0.4036 ± 0.0031 | 0.3618 |
| query pop f-LIF base (msn) | 336 | 0.2952 ± 0.0074 | 0.3967 ± 0.0029 | 0.1708 ± 0.0049 | 0.3694 ± 0.0019 | 0.3660 |
| query pop f-LIF base (msn) | 720 | 0.2906 ± 0.0017 | 0.4120 ± 0.0089 | 0.1791 ± 0.0061 | 0.3920 ± 0.0051 | 0.3843 |
| query pop f-LIF nomem (no state across tokens) | 96 | 0.2906 ± 0.0031 | 0.4156 ± 0.0045 | 0.1766 ± 0.0015 | 0.4024 ± 0.0025 | 0.3607 |
| query pop f-LIF nomem (no state across tokens) | 336 | 0.2920 ± 0.0083 | 0.4104 ± 0.0080 | 0.1749 ± 0.0053 | 0.3661 ± 0.0049 | 0.3640 |
| query pop f-LIF nomem (no state across tokens) | 720 | 0.3025 ± 0.0118 | 0.4054 ± 0.0013 | 0.1784 ± 0.0032 | 0.3945 ± 0.0229 | 0.3774 |
| query pop f-LIF nshuf (shuffled token order) | 96 | 0.2910 ± 0.0040 | 0.4160 ± 0.0036 | 0.1764 ± 0.0002 | 0.4036 ± 0.0022 | 0.3605 |
| query pop f-LIF nshuf (shuffled token order) | 336 | 0.2954 ± 0.0094 | 0.3911 ± 0.0014 | 0.1740 ± 0.0052 | 0.3649 ± 0.0045 | 0.3647 |
| query pop f-LIF nshuf (shuffled token order) | 720 | 0.2928 ± 0.0036 | 0.3998 ± 0.0020 | 0.1785 ± 0.0043 | 0.3863 ± 0.0154 | 0.3777 |

## L1 — base − Spikformer-style at each input length (r = A / B − 1 of test MSE, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B |
|---|---|---|---|---|---|---|---|---|
| L1 query pop LIF − Spikformer-style, input 96 | +0.96 (no clear difference) | -1.57 (better) | +0.08 (no clear difference) | +0.04 (no clear difference) | -0.12 | 6/12 | no clear difference | 1.25 |
| L1 query pop LIF − Spikformer-style, input 336 | -0.18 (no clear difference) | +0.11 (no clear difference) | -1.68 (better) | -1.07 (better) | -0.71 | 10/12 | no clear difference | 1.27 |
| L1 query pop LIF − Spikformer-style, input 720 | -3.55 (better) | +8.78 (worse) | +0.81 (no clear difference) | -3.96 (better) | +0.52 | 6/12 | no clear difference | 1.30 |
| L1 query pop f-LIF − Spikformer-style, input 96 | +0.04 (no clear difference) | -1.35 (better) | +0.43 (no clear difference) | +0.03 (no clear difference) | -0.21 | 6/12 | no clear difference | 1.51 |
| L1 query pop f-LIF − Spikformer-style, input 336 | -4.48 (better) | -0.38 (no clear difference) | -2.03 (better) | +0.60 (no clear difference) | -1.57 | 8/12 | better | 1.65 |
| L1 query pop f-LIF − Spikformer-style, input 720 | -2.84 (better) | +1.27 (worse) | +1.96 (worse) | +0.62 (no clear difference) | +0.25 | 5/12 | no clear difference | 1.75 |

## L2 — change of the gap with the input length (g = r_L − r_shorter per cell and seed, %p; negative = the base gains more)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean g | pairs negative | verdict |
|---|---|---|---|---|---|---|---|
| L2 query pop LIF: gap at 336 − gap at 96 | -1.15 | +1.68 | -1.76 | -1.11 | -0.58 | 8/12 | no clear change |
| L2 query pop LIF: gap at 720 − gap at 96 | -4.51 | +10.35 | +0.73 | -4.00 | +0.64 | 7/12 | no clear change |
| L2 query pop LIF: gap at 720 − gap at 336 | -3.36 | +8.67 | +2.49 | -2.89 | +1.23 | 4/12 | shrank |
| L2 query pop f-LIF: gap at 336 − gap at 96 | -4.52 | +0.97 | -2.46 | +0.57 | -1.36 | 5/12 | no clear change |
| L2 query pop f-LIF: gap at 720 − gap at 96 | -2.88 | +2.62 | +1.53 | +0.59 | +0.46 | 4/12 | no clear change |
| L2 query pop f-LIF: gap at 720 − gap at 336 | +1.64 | +1.65 | +3.99 | +0.02 | +1.83 | 4/12 | shrank |

## L3 — controls of the base model at each input length (r = A / B − 1 of test MSE, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B |
|---|---|---|---|---|---|---|---|---|
| L3 query pop LIF: msn − nshuf (order), input 96 | +0.35 (no clear difference) | +0.83 (no clear difference) | +0.57 (no clear difference) | +0.18 (no clear difference) | +0.48 | 4/12 | no clear difference | 1.00 |
| L3 query pop LIF: msn − nshuf (order), input 336 | +5.56 (worse) | +1.34 (worse) | -2.50 (better) | -2.42 (better) | +0.50 | 5/12 | no clear difference | 1.01 |
| L3 query pop LIF: msn − nshuf (order), input 720 | -1.23 (better) | +7.85 (worse) | -1.17 (better) | -1.13 (better) | +1.08 | 6/12 | no clear difference | 1.01 |
| L3 query pop f-LIF: msn − nshuf (order), input 96 | -0.19 (no clear difference) | +0.14 (no clear difference) | +1.02 (worse) | +0.00 (no clear difference) | +0.24 | 4/12 | no clear difference | 1.00 |
| L3 query pop f-LIF: msn − nshuf (order), input 336 | -0.03 (no clear difference) | +1.45 (worse) | -1.71 (better) | +1.25 (worse) | +0.24 | 3/12 | no clear difference | 1.00 |
| L3 query pop f-LIF: msn − nshuf (order), input 720 | -0.76 (no clear difference) | +3.06 (worse) | +0.38 (no clear difference) | +1.53 (worse) | +1.05 | 4/12 | worse | 1.00 |
| L3 query pop LIF: msn − nomem (memory), input 96 | +1.02 (worse) | -0.41 (no clear difference) | +0.45 (no clear difference) | +0.05 (no clear difference) | +0.28 | 4/12 | no clear difference | 1.04 |
| L3 query pop LIF: msn − nomem (memory), input 336 | +3.80 (worse) | +1.00 (worse) | -2.86 (better) | -0.76 (no clear difference) | +0.30 | 5/12 | no clear difference | 1.06 |
| L3 query pop LIF: msn − nomem (memory), input 720 | -2.75 (better) | +8.97 (worse) | -1.11 (better) | -1.71 (better) | +0.85 | 8/12 | no clear difference | 1.07 |
| L3 query pop f-LIF: msn − nomem (memory), input 96 | -0.06 (no clear difference) | +0.24 (no clear difference) | +0.93 (no clear difference) | +0.29 (no clear difference) | +0.35 | 4/12 | no clear difference | 1.22 |
| L3 query pop f-LIF: msn − nomem (memory), input 336 | +1.09 (worse) | -3.31 (better) | -2.30 (better) | +0.91 (no clear difference) | -0.90 | 7/12 | no clear difference | 1.31 |
| L3 query pop f-LIF: msn − nomem (memory), input 720 | -3.85 (better) | +1.62 (worse) | +0.43 (no clear difference) | -0.45 (no clear difference) | -0.56 | 5/12 | no clear difference | 1.39 |
| report query pop LIF: nshuf − nomem (report only), input 96 | +0.67 (no clear difference) | -1.22 (better) | -0.11 (no clear difference) | -0.13 (no clear difference) | -0.20 | 6/12 | no clear difference | 1.04 |
| report query pop LIF: nshuf − nomem (report only), input 336 | -1.64 (better) | -0.34 (no clear difference) | -0.34 (no clear difference) | +1.71 (worse) | -0.15 | 7/12 | no clear difference | 1.05 |
| report query pop LIF: nshuf − nomem (report only), input 720 | -1.53 (better) | +1.00 (no clear difference) | +0.08 (no clear difference) | -0.48 (no clear difference) | -0.23 | 8/12 | no clear difference | 1.06 |
| report query pop f-LIF: nshuf − nomem (report only), input 96 | +0.13 (no clear difference) | +0.10 (no clear difference) | -0.09 (no clear difference) | +0.29 (no clear difference) | +0.11 | 3/12 | no clear difference | 1.22 |
| report query pop f-LIF: nshuf − nomem (report only), input 336 | +1.16 (no clear difference) | -4.69 (better) | -0.42 (no clear difference) | -0.33 (no clear difference) | -1.07 | 9/12 | better | 1.31 |
| report query pop f-LIF: nshuf − nomem (report only), input 720 | -3.11 (better) | -1.40 (better) | +0.04 (no clear difference) | -1.91 (better) | -1.59 | 8/12 | better | 1.39 |

## L4 (report only) — every model against itself at the shorter input (same test targets; r = A / B − 1, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B |
|---|---|---|---|---|---|---|---|---|
| Spikformer-style: input 336 − input 96 | +6.59 (worse) | -5.69 (better) | -1.74 (better) | -8.98 (better) | -2.45 | 9/12 | better | 3.36 |
| Spikformer-style: input 720 − input 96 | +3.02 (worse) | -3.58 (better) | -1.01 (better) | -3.44 (better) | -1.25 | 8/12 | better | 7.06 |
| Spikformer-style: input 720 − input 336 | -3.21 (better) | +2.24 (worse) | +0.75 (no clear difference) | +6.10 (worse) | +1.47 | 4/12 | worse | 2.10 |
| query pop LIF base (msn): input 336 − input 96 | +5.29 (worse) | -4.08 (better) | -3.47 (better) | -9.99 (better) | -3.06 | 9/12 | better | 3.43 |
| query pop LIF base (msn): input 720 − input 96 | -1.60 (better) | +6.50 (worse) | -0.29 (no clear difference) | -7.34 (better) | -0.68 | 8/12 | no clear difference | 7.34 |
| query pop LIF base (msn): input 720 − input 336 | -6.53 (better) | +11.03 (worse) | +3.30 (worse) | +2.95 (worse) | +2.69 | 3/12 | worse | 2.14 |
| query pop LIF nomem (no state across tokens): input 336 − input 96 | +2.55 (worse) | -5.42 (better) | -0.15 (no clear difference) | -9.25 (better) | -3.07 | 8/12 | better | 3.36 |
| query pop LIF nomem (no state across tokens): input 720 − input 96 | +2.20 (worse) | -2.63 (better) | +1.29 (worse) | -5.67 (better) | -1.20 | 6/12 | no clear difference | 7.09 |
| query pop LIF nomem (no state across tokens): input 720 − input 336 | -0.27 (no clear difference) | +2.94 (worse) | +1.48 (worse) | +3.95 (worse) | +2.03 | 3/12 | worse | 2.11 |
| query pop LIF nshuf (shuffled token order): input 336 − input 96 | +0.15 (no clear difference) | -4.57 (better) | -0.37 (no clear difference) | -7.59 (better) | -3.10 | 9/12 | better | 3.40 |
| query pop LIF nshuf (shuffled token order): input 720 − input 96 | -0.04 (no clear difference) | -0.46 (no clear difference) | +1.50 (worse) | -6.01 (better) | -1.25 | 7/12 | no clear difference | 7.26 |
| query pop LIF nshuf (shuffled token order): input 720 − input 336 | -0.13 (no clear difference) | +4.31 (worse) | +1.96 (worse) | +1.74 (worse) | +1.97 | 3/12 | worse | 2.14 |
| query pop f-LIF base (msn): input 336 − input 96 | +1.66 (worse) | -4.76 (better) | -4.13 (better) | -8.46 (better) | -3.92 | 10/12 | better | 3.66 |
| query pop f-LIF base (msn): input 720 − input 96 | +0.05 (no clear difference) | -1.07 (better) | +0.52 (no clear difference) | -2.88 (better) | -0.84 | 8/12 | no clear difference | 8.14 |
| query pop f-LIF base (msn): input 720 − input 336 | -1.54 (better) | +3.86 (worse) | +4.96 (worse) | +6.11 (worse) | +3.35 | 3/12 | worse | 2.23 |
| query pop f-LIF nomem (no state across tokens): input 336 − input 96 | +0.52 (no clear difference) | -1.24 (better) | -0.99 (no clear difference) | -9.02 (better) | -2.68 | 8/12 | better | 3.39 |
| query pop f-LIF nomem (no state across tokens): input 720 − input 96 | +4.08 (worse) | -2.45 (better) | +1.04 (worse) | -1.95 (better) | +0.18 | 7/12 | no clear difference | 7.13 |
| query pop f-LIF nomem (no state across tokens): input 720 − input 336 | +3.62 (worse) | -1.18 (better) | +2.07 (worse) | +7.78 (worse) | +3.07 | 3/12 | worse | 2.10 |
| query pop f-LIF nshuf (shuffled token order): input 336 − input 96 | +1.54 (worse) | -6.00 (better) | -1.40 (better) | -9.58 (better) | -3.86 | 9/12 | better | 3.64 |
| query pop f-LIF nshuf (shuffled token order): input 720 − input 96 | +0.63 (no clear difference) | -3.91 (better) | +1.19 (worse) | -4.28 (better) | -1.59 | 7/12 | no clear difference | 8.10 |
| query pop f-LIF nshuf (shuffled token order): input 720 − input 336 | -0.77 (no clear difference) | +2.22 (worse) | +2.73 (worse) | +5.91 (worse) | +2.52 | 3/12 | worse | 2.22 |

## conclusions (rule of the pre-registration)

| comparison | pop LIF | pop f-LIF | conclusion |
|---|---|---|---|
| L1 base − Spikformer-style, input 336 | no clear difference | better | neuron-dependent (weak) |
| L1 base − Spikformer-style, input 720 | no clear difference | no clear difference | no clear difference |
| L2 gap at 336 − gap at 96 | no clear change | no clear change | no clear change |
| L2 gap at 720 − gap at 96 | no clear change | no clear change | no clear change |
| L2 gap at 720 − gap at 336 | shrank | shrank | shrank |
| L3 msn − nshuf (order), input 96 | no clear difference | no clear difference | no clear difference |
| L3 msn − nshuf (order), input 336 | no clear difference | no clear difference | no clear difference |
| L3 msn − nshuf (order), input 720 | no clear difference | worse | neuron-dependent (weak) |
| L3 msn − nomem (memory), input 96 | no clear difference | no clear difference | no clear difference |
| L3 msn − nomem (memory), input 336 | no clear difference | no clear difference | no clear difference |
| L3 msn − nomem (memory), input 720 | no clear difference | no clear difference | no clear difference |

## report only

### firing rates, energy, parameters, run time (mean of seeds)

| model | input | rho_neo (4 cells) | rho_neo mean | rho_hippo mean | energy uJ (head recount, mean of 4 cells) | energy / Spikformer-style (mean of per-cell ratios) | params ETTh2 H96 / ETTm2 H720 | run time s (min–max) |
|---|---|---|---|---|---|---|---|---|
| Spikformer-style | 96 | — | — | 0.0560 | 10.46 | 1.00 | 169656 / 1132656 | 234–859 |
| Spikformer-style | 336 | — | — | 0.0534 | 35.07 | 1.00 | 539256 / 3899376 | 161–805 |
| Spikformer-style | 720 | — | — | 0.0555 | 73.67 | 1.00 | 1130616 / 8326128 | 201–1372 |
| query pop LIF base (msn) | 96 | 0.0182 / 0.0170 / 0.0537 / 0.0422 | 0.0328 | 0.0574 | 13.04 | 1.25 | 175544 / 1143024 | 172–1179 |
| query pop LIF base (msn) | 336 | 0.0664 / 0.0573 / 0.0889 / 0.0986 | 0.0778 | 0.0550 | 44.70 | 1.27 | 545144 / 3909744 | 580–7066 |
| query pop LIF base (msn) | 720 | 0.0927 / 0.0691 / 0.1372 / 0.1257 | 0.1062 | 0.0590 | 95.58 | 1.30 | 1136504 / 8336496 | 1019–7027 |
| query pop LIF nomem (no state across tokens) | 96 | 0.0000 / 0.0000 / 0.0002 / 0.0001 | 0.0001 | 0.0511 | 12.60 | 1.20 | 175544 / 1143024 | 109–529 |
| query pop LIF nomem (no state across tokens) | 336 | 0.0000 / 0.0001 / 0.0002 / 0.0002 | 0.0001 | 0.0475 | 42.27 | 1.20 | 545144 / 3909744 | 150–1061 |
| query pop LIF nomem (no state across tokens) | 720 | 0.0000 / 0.0001 / 0.0002 / 0.0001 | 0.0001 | 0.0475 | 89.17 | 1.21 | 1136504 / 8336496 | 412–3215 |
| query pop LIF nshuf (shuffled token order) | 96 | 0.0134 / 0.0218 / 0.0526 / 0.0222 | 0.0275 | 0.0608 | 13.06 | 1.25 | 175544 / 1143024 | 195–999 |
| query pop LIF nshuf (shuffled token order) | 336 | 0.0322 / 0.0221 / 0.0443 / 0.0558 | 0.0386 | 0.0542 | 44.34 | 1.26 | 545144 / 3909744 | 373–6666 |
| query pop LIF nshuf (shuffled token order) | 720 | 0.0344 / 0.0309 / 0.0626 / 0.0762 | 0.0510 | 0.0607 | 94.73 | 1.28 | 1136504 / 8336496 | 1122–21321 |
| query pop f-LIF base (msn) | 96 | 0.0220 / 0.0194 / 0.0576 / 0.0496 | 0.0371 | 0.0581 | 15.86 | 1.51 | 175544 / 1143024 | 177–1039 |
| query pop f-LIF base (msn) | 336 | 0.0552 / 0.0496 / 0.0785 / 0.0783 | 0.0654 | 0.0529 | 57.96 | 1.65 | 545144 / 3909744 | 721–6611 |
| query pop f-LIF base (msn) | 720 | 0.0766 / 0.0669 / 0.1121 / 0.1182 | 0.0935 | 0.0589 | 129.10 | 1.75 | 1136504 / 8336496 | 1018–7613 |
| query pop f-LIF nomem (no state across tokens) | 96 | 0.0036 / 0.0034 / 0.0028 / 0.0033 | 0.0033 | 0.0540 | 12.99 | 1.24 | 175544 / 1143024 | 135–451 |
| query pop f-LIF nomem (no state across tokens) | 336 | 0.0024 / 0.0035 / 0.0037 / 0.0045 | 0.0035 | 0.0492 | 43.99 | 1.25 | 545144 / 3909744 | 175–1382 |
| query pop f-LIF nomem (no state across tokens) | 720 | 0.0023 / 0.0036 / 0.0043 / 0.0046 | 0.0037 | 0.0480 | 92.46 | 1.25 | 1136504 / 8336496 | 289–3247 |
| query pop f-LIF nshuf (shuffled token order) | 96 | 0.0131 / 0.0229 / 0.0627 / 0.0244 | 0.0308 | 0.0592 | 15.87 | 1.52 | 175544 / 1143024 | 176–1283 |
| query pop f-LIF nshuf (shuffled token order) | 336 | 0.0282 / 0.0240 / 0.0430 / 0.0424 | 0.0344 | 0.0535 | 57.72 | 1.64 | 545144 / 3909744 | 379–6742 |
| query pop f-LIF nshuf (shuffled token order) | 720 | 0.0303 / 0.0279 / 0.0697 / 0.0654 | 0.0483 | 0.0615 | 128.48 | 1.74 | 1136504 / 8336496 | 1024–20951 |

### energy by cell (uJ, head recount, mean of seeds)

| model | input | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|---|
| Spikformer-style | 96 | 9.30 | 10.79 | 10.28 | 11.48 |
| Spikformer-style | 336 | 32.26 | 35.02 | 35.63 | 37.37 |
| Spikformer-style | 720 | 68.65 | 72.58 | 74.90 | 78.53 |
| query pop LIF base (msn) | 96 | 11.30 | 13.00 | 13.35 | 14.53 |
| query pop LIF base (msn) | 336 | 39.96 | 43.29 | 46.41 | 49.16 |
| query pop LIF base (msn) | 720 | 86.35 | 90.44 | 100.98 | 104.56 |
| query pop LIF nomem (no state across tokens) | 96 | 10.86 | 12.63 | 12.92 | 14.01 |
| query pop LIF nomem (no state across tokens) | 336 | 37.83 | 40.27 | 44.61 | 46.37 |
| query pop LIF nomem (no state across tokens) | 720 | 79.86 | 85.55 | 93.45 | 97.83 |
| query pop LIF nshuf (shuffled token order) | 96 | 11.32 | 12.98 | 13.45 | 14.48 |
| query pop LIF nshuf (shuffled token order) | 336 | 39.65 | 42.56 | 46.31 | 48.82 |
| query pop LIF nshuf (shuffled token order) | 720 | 84.99 | 90.04 | 99.39 | 104.48 |
| query pop f-LIF base (msn) | 96 | 13.56 | 15.16 | 16.77 | 17.94 |
| query pop f-LIF base (msn) | 336 | 50.50 | 53.69 | 62.48 | 65.15 |
| query pop f-LIF base (msn) | 720 | 112.34 | 117.93 | 140.96 | 145.16 |
| query pop f-LIF nomem (no state across tokens) | 96 | 11.33 | 12.96 | 13.26 | 14.39 |
| query pop f-LIF nomem (no state across tokens) | 336 | 39.26 | 43.29 | 45.64 | 47.79 |
| query pop f-LIF nomem (no state across tokens) | 720 | 83.54 | 88.63 | 96.48 | 101.19 |
| query pop f-LIF nshuf (shuffled token order) | 96 | 13.55 | 15.22 | 16.86 | 17.83 |
| query pop f-LIF nshuf (shuffled token order) | 336 | 50.30 | 53.40 | 62.53 | 64.66 |
| query pop f-LIF nshuf (shuffled token order) | 720 | 112.19 | 116.48 | 140.94 | 144.30 |

### test MAE (mean of seeds)

| model | input | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|---|
| Spikformer-style | 96 | 0.3350 | 0.4366 | 0.2536 | 0.3959 |
| Spikformer-style | 336 | 0.3534 | 0.4239 | 0.2551 | 0.3844 |
| Spikformer-style | 720 | 0.3579 | 0.4354 | 0.2617 | 0.3979 |
| query pop LIF base (msn) | 96 | 0.3356 | 0.4330 | 0.2535 | 0.3954 |
| query pop LIF base (msn) | 336 | 0.3519 | 0.4253 | 0.2545 | 0.3819 |
| query pop LIF base (msn) | 720 | 0.3506 | 0.4559 | 0.2629 | 0.3909 |
| query pop LIF nomem (no state across tokens) | 96 | 0.3347 | 0.4355 | 0.2530 | 0.3951 |
| query pop LIF nomem (no state across tokens) | 336 | 0.3470 | 0.4232 | 0.2566 | 0.3831 |
| query pop LIF nomem (no state across tokens) | 720 | 0.3533 | 0.4344 | 0.2632 | 0.3993 |
| query pop LIF nshuf (shuffled token order) | 96 | 0.3361 | 0.4323 | 0.2530 | 0.3954 |
| query pop LIF nshuf (shuffled token order) | 336 | 0.3435 | 0.4225 | 0.2569 | 0.3859 |
| query pop LIF nshuf (shuffled token order) | 720 | 0.3496 | 0.4371 | 0.2631 | 0.3948 |
| query pop f-LIF base (msn) | 96 | 0.3347 | 0.4338 | 0.2540 | 0.3956 |
| query pop f-LIF base (msn) | 336 | 0.3465 | 0.4236 | 0.2540 | 0.3855 |
| query pop f-LIF base (msn) | 720 | 0.3511 | 0.4394 | 0.2645 | 0.3988 |
| query pop f-LIF nomem (no state across tokens) | 96 | 0.3350 | 0.4342 | 0.2530 | 0.3948 |
| query pop f-LIF nomem (no state across tokens) | 336 | 0.3447 | 0.4319 | 0.2562 | 0.3830 |
| query pop f-LIF nomem (no state across tokens) | 720 | 0.3561 | 0.4346 | 0.2628 | 0.4002 |
| query pop f-LIF nshuf (shuffled token order) | 96 | 0.3355 | 0.4351 | 0.2525 | 0.3953 |
| query pop f-LIF nshuf (shuffled token order) | 336 | 0.3457 | 0.4212 | 0.2550 | 0.3817 |
| query pop f-LIF nshuf (shuffled token order) | 720 | 0.3519 | 0.4307 | 0.2630 | 0.3960 |
