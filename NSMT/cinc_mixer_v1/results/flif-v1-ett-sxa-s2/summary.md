# Fusion redesign stage 2 (OT plan mask, MUX versions; LIF, ADD, one block; ETTh1 / ETTm1 x H96 / H720 x 3 seeds)

Rules: summarize_sxa2.py docstring. 36 new runs (commit `ee3fdbba6`) + stage-1 rows of S and A1 (2fed8c580).

## 4-cell means (seed means)

| label | arm | MSE | MAE | best val L1 | energy µJ (syops) | energy µJ (integer) |
|---|---|--:|--:|--:|--:|--:|
| A1 | muxlin indep. | 0.4173 | 0.4110 | 0.6138 | 16.92 | 4.09 |
| S | sxdpc indep. | 0.4194 | 0.4123 | 0.6146 | 17.27 | 4.52 |
| OT | sxdot indep. | 0.4202 | 0.4125 | 0.6144 | 18.05 | 5.19 |
| PM | msxdpc indep. | 0.4142 | 0.4094 | 0.6136 | 17.45 | 4.60 |
| OM | msxdot indep. | 0.4151 | 0.4100 | 0.6135 | 18.23 | 5.32 |

## Comparisons (test)

| kind | A | B | MSE R % | lower / higher | MSE verdict | MAE R % | MAE verdict | energy A/B syops | integer |
|---|---|---|--:|---|---|--:|---|--:|--:|
| mask OT vs Pearson | sxdot indep. | sxdpc indep. | +0.16 | 5 / 7 | **no clear difference** | +0.06 | no clear difference | 1.052 | 1.144 |
| mask OT vs Pearson | msxdot indep. | msxdpc indep. | +0.24 | 4 / 8 | **no clear difference** | +0.14 | no clear difference | 1.051 | 1.149 |
| MUX vs none | msxdpc indep. | sxdpc indep. | -1.10 | 9 / 3 | **better** | -0.62 | no clear difference | 1.015 | 1.024 |
| MUX vs none | msxdot indep. | sxdot indep. | -1.01 | 9 / 3 | **better** | -0.53 | no clear difference | 1.014 | 1.030 |
| vs muxlin | sxdot indep. | muxlin indep. | +0.59 | 6 / 6 | **no clear difference** | +0.30 | no clear difference | 1.074 | 1.252 |
| vs muxlin | msxdpc indep. | muxlin indep. | -0.70 | 6 / 6 | **no clear difference** | -0.39 | no clear difference | 1.036 | 1.118 |
| vs muxlin | msxdot indep. | muxlin indep. | -0.47 | 5 / 7 | **no clear difference** | -0.24 | no clear difference | 1.089 | 1.289 |

## Per cell (seed-mean test MSE / MAE, syops energy µJ)

| arm | ETTh1 H96 | ETTh1 H720 | ETTm1 H96 | ETTm1 H720 |
|---|---|---|---|---|
| A1 muxlin indep. | 0.3814 / 0.3930 / 18.37 | 0.4958 / 0.4687 / 33.79 | 0.3223 / 0.3468 / 3.91 | 0.4696 / 0.4355 / 11.62 |
| S sxdpc indep. | 0.3844 / 0.3940 / 18.98 | 0.5073 / 0.4774 / 34.40 | 0.3207 / 0.3452 / 4.00 | 0.4653 / 0.4327 / 11.71 |
| OT sxdot indep. | 0.3810 / 0.3924 / 20.23 | 0.5091 / 0.4773 / 35.65 | 0.3218 / 0.3471 / 4.31 | 0.4692 / 0.4331 / 12.02 |
| PM msxdpc indep. | 0.3837 / 0.3938 / 19.21 | 0.4895 / 0.4667 / 34.63 | 0.3187 / 0.3447 / 4.12 | 0.4647 / 0.4324 / 11.83 |
| OM msxdot indep. | 0.3837 / 0.3936 / 20.46 | 0.4909 / 0.4679 / 35.88 | 0.3205 / 0.3462 / 4.43 | 0.4651 / 0.4321 / 12.15 |

## Selection (validation only)

- sxdpc indep.: score 1.001407
- sxdot indep.: score 1.001730
- msxdpc indep.: score 0.999538
- msxdot indep.: score 0.999708
- **selected: msxdpc indep.** (selected_s2.json)
