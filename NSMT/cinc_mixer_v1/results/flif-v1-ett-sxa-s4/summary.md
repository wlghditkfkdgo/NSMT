# Fusion redesign stage 4: additive mask (P = A_s + alpha M, alpha learnable from 0) in Pearson / OT + MUX, IAND (LIF, one block; ETTh1 / ETTm1 x H96 / H720 x 3 seeds)

Rules: summarize_sxa4.py docstring. 24 new runs, commit `3d63b0ea2`.

## 4-cell means (seed means)

| arm | MSE | MAE | best val L1 | energy µJ (syops) | energy µJ (integer) |
|---|--:|--:|--:|--:|--:|
| Pearson add (msxdpca) | 0.4128 | 0.4076 | 0.6111 | 10.90 | 3.92 |
| OT add (msxdota) | 0.4142 | 0.4077 | 0.6110 | 11.68 | 4.71 |
| Pearson multiply (msxdpc, 3a) | 0.4137 | 0.4069 | 0.6109 | 10.92 | 3.94 |
| muxlin IAND (ref) | 0.4117 | 0.4067 | 0.6106 | 10.38 | 3.39 |

## Comparisons (test)

| kind | A | B | MSE R % | lower / higher | MSE verdict | MAE R % | MAE verdict | energy A/B syops | integer |
|---|---|---|--:|---|---|--:|---|--:|--:|
| add vs multiply | Pearson add (msxdpca) | Pearson multiply (msxdpc, 3a) | -0.20 | 7 / 5 | **no clear difference** | +0.14 | no clear difference | 0.998 | 0.996 |
| OT vs Pearson (added) | OT add (msxdota) | Pearson add (msxdpca) | +0.31 | 5 / 7 | **no clear difference** | +0.01 | no clear difference | 1.091 | 1.183 |
| vs muxlin IAND | Pearson add (msxdpca) | muxlin IAND (ref) | +0.20 | 8 / 4 | **no clear difference** | +0.19 | no clear difference | 1.066 | 1.143 |
| vs muxlin IAND | OT add (msxdota) | muxlin IAND (ref) | +0.51 | 5 / 7 | **no clear difference** | +0.20 | no clear difference | 1.165 | 1.355 |

## Per cell (seed-mean test MSE / MAE, syops energy µJ)

| arm | ETTh1 H96 | ETTh1 H720 | ETTm1 H96 | ETTm1 H720 |
|---|---|---|---|---|
| Pearson add (msxdpca) | 0.3790 / 0.3883 / 6.89 | 0.4878 / 0.4658 / 22.27 | 0.3185 / 0.3455 / 3.36 | 0.4661 / 0.4310 / 11.07 |
| OT add (msxdota) | 0.3803 / 0.3890 / 8.13 | 0.4912 / 0.4650 / 23.52 | 0.3184 / 0.3455 / 3.69 | 0.4672 / 0.4311 / 11.38 |
| Pearson multiply (msxdpc, 3a) | 0.3812 / 0.3891 / 6.92 | 0.4896 / 0.4623 / 22.30 | 0.3184 / 0.3459 / 3.37 | 0.4657 / 0.4304 / 11.07 |
| muxlin IAND (ref) | 0.3833 / 0.3907 / 6.04 | 0.4788 / 0.4590 / 21.46 | 0.3180 / 0.3448 / 3.15 | 0.4668 / 0.4324 / 10.86 |

Per-cell r (MSE, %) Pearson add (msxdpca) vs Pearson multiply (msxdpc, 3a): -0.56 -0.36 +0.03 +0.09
Per-cell r (MSE, %) OT add (msxdota) vs Pearson add (msxdpca): +0.33 +0.71 -0.04 +0.23
Per-cell r (MSE, %) Pearson add (msxdpca) vs muxlin IAND (ref): -1.11 +1.89 +0.16 -0.14
Per-cell r (MSE, %) OT add (msxdota) vs muxlin IAND (ref): -0.78 +2.59 +0.12 +0.09
