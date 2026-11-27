# Fusion redesign stage 3a: Pearson + MUX (msxdpc independent) with the IAND residual (LIF, one block; ETTh1 / ETTm1 x H96 / H720 x 3 seeds)

Rules: summarize_sxa3.py docstring. 12 new runs, commit `eb3b85c00`.

## 4-cell means (seed means)

| arm | MSE | MAE | best val L1 | energy µJ (syops) | energy µJ (integer) |
|---|--:|--:|--:|--:|--:|
| msxdpc IAND (new) | 0.4137 | 0.4069 | 0.6109 | 10.92 | 3.94 |
| muxlin IAND (ref) | 0.4117 | 0.4067 | 0.6106 | 10.38 | 3.39 |
| msxdpc ADD (stage 2) | 0.4142 | 0.4094 | 0.6136 | 17.45 | 4.60 |
| muxlin ADD (stage 1) | 0.4173 | 0.4110 | 0.6138 | 16.92 | 4.09 |

## Comparisons (test)

| kind | A | B | MSE R % | lower / higher | MSE verdict | MAE R % | MAE verdict | energy A/B syops | integer |
|---|---|---|--:|---|---|--:|---|--:|--:|
| main | msxdpc IAND (new) | muxlin IAND (ref) | +0.40 | 4 / 8 | **no clear difference** | +0.05 | no clear difference | 1.069 | 1.149 |
| residual | msxdpc IAND (new) | msxdpc ADD (stage 2) | -0.12 | 7 / 5 | **no clear difference** | -0.56 | no clear difference | 0.689 | 0.875 |
| report | msxdpc IAND (new) | muxlin ADD (stage 1) | -0.83 | 8 / 4 | **no clear difference** | -0.95 | no clear difference | 0.713 | 0.981 |

## Per cell (seed-mean test MSE / MAE, syops energy µJ)

| arm | ETTh1 H96 | ETTh1 H720 | ETTm1 H96 | ETTm1 H720 |
|---|---|---|---|---|
| msxdpc IAND (new) | 0.3812 / 0.3891 / 6.92 | 0.4896 / 0.4623 / 22.30 | 0.3184 / 0.3459 / 3.37 | 0.4657 / 0.4304 / 11.07 |
| muxlin IAND (ref) | 0.3833 / 0.3907 / 6.04 | 0.4788 / 0.4590 / 21.46 | 0.3180 / 0.3448 / 3.15 | 0.4668 / 0.4324 / 10.86 |
| msxdpc ADD (stage 2) | 0.3837 / 0.3938 / 19.21 | 0.4895 / 0.4667 / 34.63 | 0.3187 / 0.3447 / 4.12 | 0.4647 / 0.4324 / 11.83 |
| muxlin ADD (stage 1) | 0.3814 / 0.3930 / 18.37 | 0.4958 / 0.4687 / 33.79 | 0.3223 / 0.3468 / 3.91 | 0.4696 / 0.4355 / 11.62 |

Per-cell r (MSE, %) of the main comparison: -0.56 +2.25 +0.14 -0.21
