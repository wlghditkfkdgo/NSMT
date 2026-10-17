# Phase A — neocortex role (72 runs OK, 0 failed checks)

## clean test MSE (best-validation weights)

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | 0.411191 | 0.285518 | 0.401147 | 0.173929 |
| base | 0.415306 | 0.289553 | 0.401433 | 0.174387 |
| neo_lif_both | 0.419786 | 0.291403 | 0.400285 | 0.174274 |
| neo_flif_both | 0.413676 | 0.287388 | 0.401368 | 0.174612 |
| neo_lif_both@L2g0.05 | 0.420931 | 0.293565 | 0.400855 | 0.174391 |
| neo_flif_both@L2g0.05 | 0.415108 | 0.286142 | 0.401173 | 0.173846 |

## A1 ΔMSE when the neocortex output is replaced: shuffle

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | +0.25% | +0.05% | -0.03% | +0.13% |
| neo_lif_both | +0.10% | -0.22% | +0.03% | +0.07% |
| neo_flif_both | +0.18% | +0.16% | +0.04% | +0.09% |
| neo_lif_both@L2g0.05 | +0.15% | -0.27% | +0.06% | +0.15% |
| neo_flif_both@L2g0.05 | -0.22% | +0.07% | +0.01% | +0.28% |

## A1 ΔMSE when the neocortex output is replaced: ones

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | +0.74% | -1.05% | +1.11% | +4.43% |
| neo_lif_both | +1.23% | -2.24% | +2.85% | +6.61% |
| neo_flif_both | +2.80% | -1.26% | +2.65% | +4.71% |
| neo_lif_both@L2g0.05 | +1.29% | +0.22% | +4.18% | +6.42% |
| neo_flif_both@L2g0.05 | +1.04% | +4.19% | +1.21% | +7.36% |

## A1 ΔMSE when the neocortex output is replaced: zeros

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | +2.09% | -0.17% | +0.49% | +16.85% |
| neo_lif_both | +0.77% | -0.75% | +0.66% | +12.04% |
| neo_flif_both | +5.74% | +0.11% | +0.40% | +11.06% |
| neo_lif_both@L2g0.05 | +0.49% | -2.81% | +1.19% | +23.68% |
| neo_flif_both@L2g0.05 | +0.15% | +0.66% | +0.69% | +16.50% |

## A1 ΔMSE when the neocortex output is replaced: permute

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | +2.21% | +0.45% | +0.18% | +2.21% |
| neo_lif_both | +1.76% | -1.32% | +0.54% | +2.14% |
| neo_flif_both | +4.31% | +1.34% | +0.36% | +2.67% |
| neo_lif_both@L2g0.05 | -0.15% | -1.92% | +0.90% | +2.05% |
| neo_flif_both@L2g0.05 | +0.51% | +0.11% | +0.25% | +2.32% |

## A1 ΔMSE when the neocortex output is replaced: reverse

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | +0.50% | -0.05% | -0.11% | -0.24% |
| neo_lif_both | +0.20% | +0.46% | +0.15% | +0.26% |
| neo_flif_both | +0.72% | -0.08% | -0.16% | +0.27% |
| neo_lif_both@L2g0.05 | +1.16% | +0.56% | +0.38% | +1.54% |
| neo_flif_both@L2g0.05 | +0.59% | +0.10% | +0.10% | +0.82% |

## A2 neo_rate

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | 0.143 | 0.135 | 0.157 | 0.190 |
| neo_lif_both | 0.153 | 0.150 | 0.152 | 0.182 |
| neo_flif_both | 0.142 | 0.132 | 0.143 | 0.173 |
| neo_lif_both@L2g0.05 | 0.138 | 0.134 | 0.182 | 0.209 |
| neo_flif_both@L2g0.05 | 0.103 | 0.094 | 0.165 | 0.199 |

## A2 hip_rate

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | 0.130 | 0.138 | 0.136 | 0.164 |
| neo_lif_both | 0.128 | 0.136 | 0.135 | 0.166 |
| neo_flif_both | 0.131 | 0.137 | 0.138 | 0.163 |
| neo_lif_both@L2g0.05 | 0.131 | 0.141 | 0.131 | 0.168 |
| neo_flif_both@L2g0.05 | 0.128 | 0.131 | 0.134 | 0.165 |

## A2 survive

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | 0.197 | 0.214 | 0.249 | 0.297 |
| neo_lif_both | 0.224 | 0.238 | 0.247 | 0.252 |
| neo_flif_both | 0.188 | 0.203 | 0.227 | 0.234 |
| neo_lif_both@L2g0.05 | 0.162 | 0.181 | 0.231 | 0.254 |
| neo_flif_both@L2g0.05 | 0.130 | 0.139 | 0.214 | 0.243 |

## A2 agree

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | 0.881 | 0.887 | 0.896 | 0.884 |
| neo_lif_both | 0.891 | 0.892 | 0.915 | 0.902 |
| neo_flif_both | 0.896 | 0.901 | 0.922 | 0.906 |
| neo_lif_both@L2g0.05 | 0.860 | 0.872 | 0.883 | 0.870 |
| neo_flif_both@L2g0.05 | 0.873 | 0.892 | 0.882 | 0.863 |

## A2 agree_chance

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | 0.755 | 0.767 | 0.736 | 0.693 |
| neo_lif_both | 0.742 | 0.746 | 0.742 | 0.703 |
| neo_flif_both | 0.756 | 0.772 | 0.756 | 0.714 |
| neo_lif_both@L2g0.05 | 0.763 | 0.768 | 0.705 | 0.670 |
| neo_flif_both@L2g0.05 | 0.815 | 0.829 | 0.727 | 0.682 |

## A4 ΔMSE with the repetition axis cut to k = 1 (of 24)

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | +2.82% | +3.93% | +0.66% | +3.23% |
| base | +2.85% | +3.13% | +0.78% | +2.46% |
| neo_lif_both | +1.98% | +3.24% | +1.00% | +2.96% |
| neo_flif_both | +4.58% | +3.13% | +0.83% | +2.85% |
| neo_lif_both@L2g0.05 | +1.52% | +0.73% | +0.82% | +3.11% |
| neo_flif_both@L2g0.05 | +1.96% | +3.17% | +1.00% | +3.01% |

## A4 ΔMSE with the repetition axis cut to k = 2 (of 24)

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | +0.68% | +0.37% | +0.13% | +1.12% |
| base | +1.42% | +0.99% | +0.19% | +0.55% |
| neo_lif_both | +0.79% | +0.41% | +0.29% | +0.69% |
| neo_flif_both | +1.31% | +0.60% | +0.16% | +0.66% |
| neo_lif_both@L2g0.05 | -0.37% | -0.21% | +0.21% | +0.49% |
| neo_flif_both@L2g0.05 | +0.12% | +0.50% | +0.22% | +0.49% |

## A4 ΔMSE with the repetition axis cut to k = 4 (of 24)

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | +0.14% | +0.06% | +0.08% | +0.42% |
| base | +0.47% | +0.27% | +0.11% | +0.07% |
| neo_lif_both | +0.03% | -0.16% | +0.15% | +0.09% |
| neo_flif_both | +0.26% | +0.04% | +0.08% | +0.12% |
| neo_lif_both@L2g0.05 | -0.69% | -0.48% | +0.10% | +0.00% |
| neo_flif_both@L2g0.05 | -0.41% | +0.05% | +0.06% | +0.02% |

## A4 ΔMSE with the repetition axis cut to k = 8 (of 24)

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | -0.01% | -0.07% | -0.00% | +0.14% |
| base | +0.14% | +0.09% | +0.00% | -0.01% |
| neo_lif_both | -0.01% | -0.05% | +0.03% | -0.01% |
| neo_flif_both | +0.09% | -0.00% | -0.00% | +0.03% |
| neo_lif_both@L2g0.05 | -0.27% | -0.17% | -0.01% | -0.04% |
| neo_flif_both@L2g0.05 | -0.15% | -0.01% | -0.01% | -0.04% |

## A4 ΔMSE with the repetition axis cut to k = 12 (of 24)

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | -0.04% | -0.04% | +0.00% | +0.04% |
| base | +0.00% | -0.01% | +0.01% | -0.03% |
| neo_lif_both | -0.07% | -0.08% | +0.02% | -0.04% |
| neo_flif_both | +0.02% | -0.03% | +0.01% | -0.02% |
| neo_lif_both@L2g0.05 | -0.16% | -0.13% | -0.01% | -0.04% |
| neo_flif_both@L2g0.05 | -0.10% | -0.03% | -0.01% | -0.05% |

## A5 neocortex share of the input-gradient norm

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | 0.316 | 0.205 | 0.306 | 0.360 |
| neo_lif_both | 0.480 | 0.390 | 0.478 | 0.493 |
| neo_flif_both | 0.401 | 0.375 | 0.484 | 0.398 |
| neo_lif_both@L2g0.05 | 0.617 | 0.510 | 0.672 | 0.668 |
| neo_flif_both@L2g0.05 | 0.587 | 0.455 | 0.506 | 0.509 |

## A3 R² (val) raw → fut_mean

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | -7.933 | -0.550 | -0.489 | -0.016 |
| base | -7.933 | -0.550 | -0.489 | -0.016 |
| neo_lif_both | -7.933 | -0.550 | -0.489 | -0.016 |
| neo_flif_both | -7.933 | -0.550 | -0.489 | -0.016 |
| neo_lif_both@L2g0.05 | -7.933 | -0.550 | -0.489 | -0.016 |
| neo_flif_both@L2g0.05 | -7.933 | -0.550 | -0.489 | -0.016 |

## A3 R² (val) raw → fut_slope

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | -10.426 | -0.412 | -1.448 | 0.025 |
| base | -10.426 | -0.412 | -1.448 | 0.025 |
| neo_lif_both | -10.426 | -0.412 | -1.448 | 0.025 |
| neo_flif_both | -10.426 | -0.412 | -1.448 | 0.025 |
| neo_lif_both@L2g0.05 | -10.426 | -0.412 | -1.448 | 0.025 |
| neo_flif_both@L2g0.05 | -10.426 | -0.412 | -1.448 | 0.025 |

## A3 R² (val) raw → lag1

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | 1.000 | 1.000 | 1.000 | 1.000 |
| base | 1.000 | 1.000 | 1.000 | 1.000 |
| neo_lif_both | 1.000 | 1.000 | 1.000 | 1.000 |
| neo_flif_both | 1.000 | 1.000 | 1.000 | 1.000 |
| neo_lif_both@L2g0.05 | 1.000 | 1.000 | 1.000 | 1.000 |
| neo_flif_both@L2g0.05 | 1.000 | 1.000 | 1.000 | 1.000 |

## A3 R² (val) raw → lag4

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | 1.000 | 1.000 | 1.000 | 1.000 |
| base | 1.000 | 1.000 | 1.000 | 1.000 |
| neo_lif_both | 1.000 | 1.000 | 1.000 | 1.000 |
| neo_flif_both | 1.000 | 1.000 | 1.000 | 1.000 |
| neo_lif_both@L2g0.05 | 1.000 | 1.000 | 1.000 | 1.000 |
| neo_flif_both@L2g0.05 | 1.000 | 1.000 | 1.000 | 1.000 |

## A3 R² (val) raw → lag12

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | 1.000 | 1.000 | 1.000 | 1.000 |
| base | 1.000 | 1.000 | 1.000 | 1.000 |
| neo_lif_both | 1.000 | 1.000 | 1.000 | 1.000 |
| neo_flif_both | 1.000 | 1.000 | 1.000 | 1.000 |
| neo_lif_both@L2g0.05 | 1.000 | 1.000 | 1.000 | 1.000 |
| neo_flif_both@L2g0.05 | 1.000 | 1.000 | 1.000 | 1.000 |

## A3 R² (val) raw → lag23

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | 1.000 | 1.000 | 1.000 | 1.000 |
| base | 1.000 | 1.000 | 1.000 | 1.000 |
| neo_lif_both | 1.000 | 1.000 | 1.000 | 1.000 |
| neo_flif_both | 1.000 | 1.000 | 1.000 | 1.000 |
| neo_lif_both@L2g0.05 | 1.000 | 1.000 | 1.000 | 1.000 |
| neo_flif_both@L2g0.05 | 1.000 | 1.000 | 1.000 | 1.000 |

## A3 R² (val) hip_all → fut_mean

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | -24.229 | -2.092 | -2.093 | -0.156 |
| base | -24.472 | -2.132 | -2.106 | -0.159 |
| neo_lif_both | -24.835 | -2.140 | -2.148 | -0.158 |
| neo_flif_both | -25.249 | -2.130 | -2.144 | -0.161 |
| neo_lif_both@L2g0.05 | -24.957 | -2.155 | -2.014 | -0.159 |
| neo_flif_both@L2g0.05 | -25.039 | -2.075 | -2.003 | -0.159 |

## A3 R² (val) hip_all → fut_slope

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | -28.219 | -2.212 | -6.023 | -0.052 |
| base | -28.089 | -2.244 | -6.084 | -0.049 |
| neo_lif_both | -28.604 | -2.262 | -6.251 | -0.055 |
| neo_flif_both | -29.107 | -2.255 | -6.202 | -0.053 |
| neo_lif_both@L2g0.05 | -28.877 | -2.252 | -5.709 | -0.055 |
| neo_flif_both@L2g0.05 | -28.923 | -2.145 | -5.673 | -0.050 |

## A3 R² (val) hip_all → lag1

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | 0.976 | 0.984 | 0.984 | 0.994 |
| base | 0.987 | 0.984 | 0.985 | 0.994 |
| neo_lif_both | 0.985 | 0.985 | 0.982 | 0.993 |
| neo_flif_both | 0.987 | 0.985 | 0.982 | 0.994 |
| neo_lif_both@L2g0.05 | 0.986 | 0.986 | 0.989 | 0.994 |
| neo_flif_both@L2g0.05 | 0.986 | 0.983 | 0.987 | 0.995 |

## A3 R² (val) hip_all → lag4

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | 0.979 | 0.985 | 0.989 | 0.995 |
| base | 0.989 | 0.986 | 0.989 | 0.996 |
| neo_lif_both | 0.988 | 0.986 | 0.988 | 0.995 |
| neo_flif_both | 0.989 | 0.986 | 0.987 | 0.995 |
| neo_lif_both@L2g0.05 | 0.988 | 0.987 | 0.992 | 0.995 |
| neo_flif_both@L2g0.05 | 0.989 | 0.984 | 0.990 | 0.996 |

## A3 R² (val) hip_all → lag12

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | 0.972 | 0.981 | 0.989 | 0.995 |
| base | 0.985 | 0.982 | 0.989 | 0.995 |
| neo_lif_both | 0.984 | 0.982 | 0.988 | 0.995 |
| neo_flif_both | 0.985 | 0.982 | 0.987 | 0.995 |
| neo_lif_both@L2g0.05 | 0.985 | 0.983 | 0.992 | 0.995 |
| neo_flif_both@L2g0.05 | 0.985 | 0.979 | 0.990 | 0.996 |

## A3 R² (val) hip_all → lag23

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | 0.976 | 0.980 | 0.982 | 0.990 |
| base | 0.983 | 0.982 | 0.983 | 0.991 |
| neo_lif_both | 0.982 | 0.981 | 0.979 | 0.989 |
| neo_flif_both | 0.983 | 0.982 | 0.979 | 0.990 |
| neo_lif_both@L2g0.05 | 0.983 | 0.982 | 0.987 | 0.990 |
| neo_flif_both@L2g0.05 | 0.983 | 0.980 | 0.984 | 0.991 |

## A3 R² (val) hip_last → fut_mean

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | -18.780 | -0.898 | -1.404 | -0.136 |
| base | -24.252 | -0.906 | -1.438 | -0.135 |
| neo_lif_both | -22.586 | -0.905 | -1.306 | -0.138 |
| neo_flif_both | -24.038 | -0.911 | -1.311 | -0.133 |
| neo_lif_both@L2g0.05 | -24.074 | -0.919 | -1.557 | -0.130 |
| neo_flif_both@L2g0.05 | -24.031 | -0.909 | -1.458 | -0.132 |

## A3 R² (val) hip_last → fut_slope

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | -24.474 | -1.234 | -4.411 | -0.079 |
| base | -31.466 | -1.284 | -4.491 | -0.080 |
| neo_lif_both | -29.398 | -1.320 | -4.088 | -0.082 |
| neo_flif_both | -31.216 | -1.280 | -4.118 | -0.079 |
| neo_lif_both@L2g0.05 | -31.296 | -1.347 | -4.887 | -0.081 |
| neo_flif_both@L2g0.05 | -31.230 | -1.156 | -4.538 | -0.078 |

## A3 R² (val) hip_last → lag1

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | 0.789 | 0.802 | 0.874 | 0.886 |
| base | 0.812 | 0.805 | 0.874 | 0.887 |
| neo_lif_both | 0.806 | 0.804 | 0.869 | 0.888 |
| neo_flif_both | 0.811 | 0.806 | 0.870 | 0.887 |
| neo_lif_both@L2g0.05 | 0.807 | 0.806 | 0.879 | 0.884 |
| neo_flif_both@L2g0.05 | 0.809 | 0.798 | 0.874 | 0.887 |

## A3 R² (val) hip_last → lag4

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | 0.009 | 0.007 | 0.158 | 0.161 |
| base | 0.007 | 0.006 | 0.158 | 0.161 |
| neo_lif_both | 0.008 | 0.007 | 0.152 | 0.162 |
| neo_flif_both | 0.006 | 0.007 | 0.154 | 0.161 |
| neo_lif_both@L2g0.05 | 0.006 | 0.006 | 0.156 | 0.156 |
| neo_flif_both@L2g0.05 | 0.008 | 0.008 | 0.156 | 0.157 |

## A3 R² (val) hip_last → lag12

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | 0.041 | 0.033 | 0.202 | 0.203 |
| base | 0.042 | 0.034 | 0.201 | 0.202 |
| neo_lif_both | 0.043 | 0.033 | 0.201 | 0.202 |
| neo_flif_both | 0.040 | 0.034 | 0.199 | 0.203 |
| neo_lif_both@L2g0.05 | 0.041 | 0.032 | 0.204 | 0.203 |
| neo_flif_both@L2g0.05 | 0.043 | 0.033 | 0.201 | 0.201 |

## A3 R² (val) hip_last → lag23

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | 0.065 | 0.071 | 0.041 | 0.045 |
| base | 0.078 | 0.073 | 0.040 | 0.048 |
| neo_lif_both | 0.076 | 0.074 | 0.040 | 0.045 |
| neo_flif_both | 0.080 | 0.073 | 0.039 | 0.046 |
| neo_lif_both@L2g0.05 | 0.077 | 0.073 | 0.041 | 0.044 |
| neo_flif_both@L2g0.05 | 0.074 | 0.071 | 0.041 | 0.046 |

## A3 R² (val) neo_all → fut_mean

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | -14.673 | -1.292 | -1.825 | -0.195 |
| neo_lif_both | -18.129 | -1.763 | -1.959 | -0.184 |
| neo_flif_both | -18.632 | -1.615 | -2.068 | -0.187 |
| neo_lif_both@L2g0.05 | -14.499 | -1.310 | -1.893 | -0.160 |
| neo_flif_both@L2g0.05 | -15.351 | -1.072 | -1.767 | -0.161 |

## A3 R² (val) neo_all → fut_slope

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | -14.751 | -1.457 | -5.008 | -0.074 |
| neo_lif_both | -19.376 | -1.821 | -5.454 | -0.070 |
| neo_flif_both | -20.210 | -1.703 | -5.753 | -0.074 |
| neo_lif_both@L2g0.05 | -14.402 | -1.403 | -5.119 | -0.040 |
| neo_flif_both@L2g0.05 | -15.555 | -1.458 | -4.699 | -0.039 |

## A3 R² (val) neo_all → lag1

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | 0.980 | 0.976 | 0.981 | 0.985 |
| neo_lif_both | 0.986 | 0.982 | 0.985 | 0.988 |
| neo_flif_both | 0.988 | 0.986 | 0.986 | 0.989 |
| neo_lif_both@L2g0.05 | 0.955 | 0.954 | 0.940 | 0.942 |
| neo_flif_both@L2g0.05 | 0.950 | 0.945 | 0.944 | 0.951 |

## A3 R² (val) neo_all → lag4

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | 0.978 | 0.972 | 0.981 | 0.987 |
| neo_lif_both | 0.985 | 0.980 | 0.985 | 0.988 |
| neo_flif_both | 0.987 | 0.984 | 0.985 | 0.990 |
| neo_lif_both@L2g0.05 | 0.952 | 0.949 | 0.949 | 0.949 |
| neo_flif_both@L2g0.05 | 0.947 | 0.941 | 0.950 | 0.955 |

## A3 R² (val) neo_all → lag12

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | 0.971 | 0.964 | 0.980 | 0.986 |
| neo_lif_both | 0.979 | 0.972 | 0.983 | 0.986 |
| neo_flif_both | 0.982 | 0.978 | 0.984 | 0.989 |
| neo_lif_both@L2g0.05 | 0.939 | 0.936 | 0.946 | 0.944 |
| neo_flif_both@L2g0.05 | 0.930 | 0.921 | 0.945 | 0.950 |

## A3 R² (val) neo_all → lag23

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | 0.970 | 0.963 | 0.969 | 0.981 |
| neo_lif_both | 0.975 | 0.968 | 0.971 | 0.978 |
| neo_flif_both | 0.978 | 0.972 | 0.970 | 0.982 |
| neo_lif_both@L2g0.05 | 0.926 | 0.923 | 0.941 | 0.941 |
| neo_flif_both@L2g0.05 | 0.906 | 0.896 | 0.931 | 0.938 |

## A3 R² (val) neo_last → fut_mean

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | -24.568 | -0.762 | -1.424 | -0.163 |
| neo_lif_both | -35.215 | -0.852 | -1.707 | -0.153 |
| neo_flif_both | -38.549 | -0.776 | -1.634 | -0.152 |
| neo_lif_both@L2g0.05 | -39.713 | -0.755 | -1.976 | -0.145 |
| neo_flif_both@L2g0.05 | -31.255 | -0.693 | -1.918 | -0.156 |

## A3 R² (val) neo_last → fut_slope

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | -28.534 | -1.713 | -4.518 | -0.091 |
| neo_lif_both | -44.834 | -1.674 | -5.444 | -0.076 |
| neo_flif_both | -48.513 | -2.111 | -5.201 | -0.082 |
| neo_lif_both@L2g0.05 | -50.059 | -2.200 | -6.108 | -0.064 |
| neo_flif_both@L2g0.05 | -37.789 | -1.743 | -5.791 | -0.072 |

## A3 R² (val) neo_last → lag1

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | 0.829 | 0.823 | 0.900 | 0.924 |
| neo_lif_both | 0.914 | 0.905 | 0.941 | 0.944 |
| neo_flif_both | 0.927 | 0.918 | 0.944 | 0.953 |
| neo_lif_both@L2g0.05 | 0.833 | 0.837 | 0.844 | 0.827 |
| neo_flif_both@L2g0.05 | 0.817 | 0.806 | 0.852 | 0.853 |

## A3 R² (val) neo_last → lag4

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | 0.033 | 0.029 | 0.263 | 0.319 |
| neo_lif_both | 0.150 | 0.143 | 0.519 | 0.501 |
| neo_flif_both | 0.028 | 0.023 | 0.274 | 0.315 |
| neo_lif_both@L2g0.05 | 0.165 | 0.182 | 0.667 | 0.687 |
| neo_flif_both@L2g0.05 | 0.038 | 0.020 | 0.378 | 0.425 |

## A3 R² (val) neo_last → lag12

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | 0.062 | 0.053 | 0.223 | 0.226 |
| neo_lif_both | 0.137 | 0.116 | 0.247 | 0.238 |
| neo_flif_both | 0.071 | 0.066 | 0.220 | 0.221 |
| neo_lif_both@L2g0.05 | 0.147 | 0.134 | 0.246 | 0.235 |
| neo_flif_both@L2g0.05 | 0.087 | 0.064 | 0.267 | 0.269 |

## A3 R² (val) neo_last → lag23

| cond | ETTh2_H720 | ETTh2_H96 | ETTm2_H720 | ETTm2_H96 |
|---|---|---|---|---|
| hippo_only | – | – | – | – |
| base | 0.117 | 0.098 | 0.037 | 0.043 |
| neo_lif_both | 0.195 | 0.173 | 0.056 | 0.060 |
| neo_flif_both | 0.161 | 0.143 | 0.047 | 0.062 |
| neo_lif_both@L2g0.05 | 0.210 | 0.191 | 0.107 | 0.137 |
| neo_flif_both@L2g0.05 | 0.148 | 0.130 | 0.208 | 0.228 |

