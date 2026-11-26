# Fusion redesign stage 1 (LIF, ADD, one block; ETTh1 / ETTm1 x H96 / H720 x 3 seeds)

Rules: summarize_sxa.py docstring. 120 runs, training commit `2fed8c580`. Energy = syops (judging rule), integer rule reported.

## 4-cell means (seed means)

| arm | MSE | MAE | best val L1 | energy µJ (syops) | energy µJ (integer) |
|---|--:|--:|--:|--:|--:|
| muxlin indep. | 0.4173 | 0.4110 | 0.6138 | 16.92 | 4.09 |
| muxlin shared | 0.4164 | 0.4115 | 0.6146 | 16.93 | 4.23 |
| sxd indep. | 0.4190 | 0.4123 | 0.6147 | 16.70 | 3.97 |
| sxd shared | 0.4160 | 0.4105 | 0.6143 | 16.70 | 3.93 |
| sxdpc indep. | 0.4194 | 0.4123 | 0.6146 | 17.27 | 4.52 |
| sxdpc shared | 0.4168 | 0.4109 | 0.6145 | 17.27 | 4.52 |
| sxp indep. | 0.4171 | 0.4116 | 0.6155 | 17.25 | 4.49 |
| sxp shared | 0.4155 | 0.4107 | 0.6150 | 17.26 | 4.48 |
| sxppc indep. | 0.4178 | 0.4121 | 0.6150 | 17.83 | 5.05 |
| sxppc shared | 0.4145 | 0.4102 | 0.6151 | 17.83 | 5.02 |
| IAND ref | 0.4117 | 0.4067 | 0.6106 | 10.38 | — |

## Comparisons (test; R = mean over cells of the seed-mean % change, A vs B)

| kind | A | B | MSE R % | lower / higher | MSE verdict | MAE R % | MAE verdict | energy A/B syops | integer |
|---|---|---|--:|---|---|--:|---|--:|--:|
| main | sxd indep. | muxlin indep. | +0.36 | 3 / 9 | **no clear difference** | +0.25 | no clear difference | 0.980 | 0.959 |
| main | sxd shared | muxlin indep. | -0.36 | 6 / 6 | **no clear difference** | -0.15 | no clear difference | 0.980 | 0.953 |
| main | sxdpc indep. | muxlin indep. | +0.44 | 5 / 7 | **no clear difference** | +0.25 | no clear difference | 1.021 | 1.091 |
| main | sxdpc shared | muxlin indep. | -0.21 | 6 / 6 | **no clear difference** | -0.04 | no clear difference | 1.021 | 1.093 |
| main | sxp indep. | muxlin indep. | +0.10 | 6 / 6 | **no clear difference** | +0.15 | no clear difference | 1.028 | 1.096 |
| main | sxp shared | muxlin indep. | -0.44 | 8 / 4 | **no clear difference** | -0.13 | no clear difference | 1.029 | 1.096 |
| main | sxppc indep. | muxlin indep. | +0.12 | 5 / 7 | **no clear difference** | +0.22 | no clear difference | 1.069 | 1.231 |
| main | sxppc shared | muxlin indep. | -0.58 | 7 / 5 | **no clear difference** | -0.21 | no clear difference | 1.070 | 1.226 |
| mask | sxdpc indep. | sxd indep. | +0.09 | 6 / 6 | **no clear difference** | -0.00 | no clear difference | 1.043 | 1.139 |
| mask | sxdpc shared | sxd shared | +0.15 | 5 / 7 | **no clear difference** | +0.11 | no clear difference | 1.042 | 1.147 |
| mask | sxppc indep. | sxp indep. | +0.03 | 8 / 4 | **no clear difference** | +0.07 | no clear difference | 1.040 | 1.123 |
| mask | sxppc shared | sxp shared | -0.13 | 5 / 7 | **no clear difference** | -0.08 | no clear difference | 1.039 | 1.118 |
| projection | sxp indep. | sxd indep. | -0.24 | 5 / 7 | **no clear difference** | -0.10 | no clear difference | 1.050 | 1.145 |
| projection | sxp shared | sxd shared | -0.09 | 5 / 7 | **no clear difference** | +0.02 | no clear difference | 1.050 | 1.151 |
| projection | sxppc indep. | sxdpc indep. | -0.30 | 6 / 6 | **no clear difference** | -0.03 | no clear difference | 1.047 | 1.128 |
| projection | sxppc shared | sxdpc shared | -0.35 | 7 / 5 | **no clear difference** | -0.16 | no clear difference | 1.048 | 1.122 |
| sharing | muxlin shared | muxlin indep. | -0.17 | 5 / 7 | **no clear difference** | +0.12 | no clear difference | 1.001 | 1.028 |
| sharing | sxd shared | sxd indep. | -0.70 | 10 / 2 | **no clear difference** | -0.40 | no clear difference | 1.000 | 0.994 |
| sharing | sxdpc shared | sxdpc indep. | -0.63 | 7 / 5 | **no clear difference** | -0.28 | no clear difference | 1.000 | 1.002 |
| sharing | sxp shared | sxp indep. | -0.53 | 7 / 5 | **no clear difference** | -0.27 | no clear difference | 1.001 | 1.000 |
| sharing | sxppc shared | sxppc indep. | -0.69 | 7 / 5 | **no clear difference** | -0.41 | no clear difference | 1.000 | 0.996 |
| ADD | muxlin indep. | IAND ref | +1.26 | 4 / 8 | **worse** | +1.01 | worse | 1.731 | — |

## Per cell (seed-mean test MSE / MAE, syops energy µJ)

| arm | ETTh1 H96 | ETTh1 H720 | ETTm1 H96 | ETTm1 H720 |
|---|---|---|---|---|
| muxlin indep. | 0.3814 / 0.3930 / 18.37 | 0.4958 / 0.4687 / 33.79 | 0.3223 / 0.3468 / 3.91 | 0.4696 / 0.4355 / 11.62 |
| muxlin shared | 0.3856 / 0.3958 / 18.37 | 0.4945 / 0.4693 / 33.78 | 0.3203 / 0.3466 / 3.91 | 0.4652 / 0.4342 / 11.64 |
| sxd indep. | 0.3837 / 0.3943 / 18.09 | 0.5035 / 0.4764 / 33.51 | 0.3212 / 0.3459 / 3.73 | 0.4677 / 0.4324 / 11.45 |
| sxd shared | 0.3836 / 0.3934 / 18.10 | 0.4964 / 0.4709 / 33.50 | 0.3178 / 0.3447 / 3.74 | 0.4661 / 0.4331 / 11.45 |
| sxdpc indep. | 0.3844 / 0.3940 / 18.98 | 0.5073 / 0.4774 / 34.40 | 0.3207 / 0.3452 / 4.00 | 0.4653 / 0.4327 / 11.71 |
| sxdpc shared | 0.3807 / 0.3929 / 18.99 | 0.5005 / 0.4724 / 34.39 | 0.3184 / 0.3459 / 4.00 | 0.4674 / 0.4326 / 11.72 |
| sxp indep. | 0.3878 / 0.3946 / 18.83 | 0.4936 / 0.4728 / 34.23 | 0.3237 / 0.3471 / 4.13 | 0.4633 / 0.4320 / 11.83 |
| sxp shared | 0.3853 / 0.3935 / 18.82 | 0.4946 / 0.4725 / 34.23 | 0.3174 / 0.3442 / 4.13 | 0.4646 / 0.4325 / 11.85 |
| sxppc indep. | 0.3840 / 0.3932 / 19.71 | 0.4978 / 0.4758 / 35.13 | 0.3214 / 0.3459 / 4.39 | 0.4679 / 0.4335 / 12.10 |
| sxppc shared | 0.3840 / 0.3937 / 19.71 | 0.4904 / 0.4695 / 35.12 | 0.3202 / 0.3455 / 4.39 | 0.4634 / 0.4321 / 12.11 |
| IAND ref | 0.3833 / 0.3907 / 6.04 | 0.4788 / 0.4590 / 21.46 | 0.3180 / 0.3448 / 3.15 | 0.4668 / 0.4324 / 10.86 |

## Selection for stage 2 (validation only)

- sxdpc indep.: score 1.001407
- sxdpc shared: score 1.001625
- sxppc indep.: score 1.002121
- sxppc shared: score 1.002018
- **selected: sxdpc indep.** (selected_s1.json)

Report only: muxlin's uncounted |H_f - H_r| per window: ETTh1 0.0194 µJ, ETTm1 0.0097 µJ.
