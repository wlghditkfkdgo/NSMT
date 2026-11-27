# T test: repetition steps T 8 vs 4 for muxlin IAND and Pearson add + MUX IAND (LIF, one block; ETTh1 / ETTm1 x H96 / H720 x 3 seeds)

Rules: summarize_t8.py docstring. 24 new runs, commit `5d88a7205`.

## 4-cell means (seed means)

| arm | MSE | MAE | best val L1 | energy µJ (syops) | energy µJ (integer) |
|---|--:|--:|--:|--:|--:|
| muxlin IAND T8 | 0.4126 | 0.4067 | 0.6107 | 13.27 | 6.95 |
| Pearson add + MUX IAND T8 | 0.4140 | 0.4067 | 0.6105 | 13.78 | 7.49 |
| muxlin IAND T4 (ref) | 0.4117 | 0.4067 | 0.6106 | 10.38 | 3.39 |
| Pearson add + MUX IAND T4 (stage 4) | 0.4128 | 0.4076 | 0.6111 | 10.90 | 3.92 |

## Comparisons (test)

| kind | A | B | MSE R % | lower / higher | MSE verdict | MAE R % | MAE verdict | energy A/B syops | integer |
|---|---|---|--:|---|---|--:|---|--:|--:|
| T 8 vs 4 | muxlin IAND T8 | muxlin IAND T4 (ref) | +0.16 | 6 / 6 | **no clear difference** | -0.01 | no clear difference | 1.406 | 2.044 |
| T 8 vs 4 | Pearson add + MUX IAND T8 | Pearson add + MUX IAND T4 (stage 4) | +0.25 | 5 / 7 | **no clear difference** | -0.21 | no clear difference | 1.372 | 1.913 |
| structure at T 8 | Pearson add + MUX IAND T8 | muxlin IAND T8 | +0.28 | 5 / 7 | **no clear difference** | -0.02 | no clear difference | 1.043 | 1.069 |

## Per cell (seed-mean test MSE / MAE, syops energy µJ)

| arm | ETTh1 H96 | ETTh1 H720 | ETTm1 H96 | ETTm1 H720 |
|---|---|---|---|---|
| muxlin IAND T8 | 0.3807 / 0.3898 / 9.86 | 0.4849 / 0.4603 / 25.23 | 0.3185 / 0.3447 / 5.14 | 0.4663 / 0.4322 / 12.86 |
| Pearson add + MUX IAND T8 | 0.3803 / 0.3887 / 10.72 | 0.4866 / 0.4618 / 26.06 | 0.3180 / 0.3440 / 5.33 | 0.4710 / 0.4326 / 13.02 |
| muxlin IAND T4 (ref) | 0.3833 / 0.3907 / 6.04 | 0.4788 / 0.4590 / 21.46 | 0.3180 / 0.3448 / 3.15 | 0.4668 / 0.4324 / 10.86 |
| Pearson add + MUX IAND T4 (stage 4) | 0.3790 / 0.3883 / 6.89 | 0.4878 / 0.4658 / 22.27 | 0.3185 / 0.3455 / 3.36 | 0.4661 / 0.4310 / 11.07 |

Per-cell r (MSE, %) muxlin IAND T8 vs muxlin IAND T4 (ref): -0.68 +1.25 +0.18 -0.09
Per-cell r (MSE, %) Pearson add + MUX IAND T8 vs Pearson add + MUX IAND T4 (stage 4): +0.33 -0.22 -0.15 +1.05
Per-cell r (MSE, %) Pearson add + MUX IAND T8 vs muxlin IAND T8: -0.10 +0.37 -0.17 +1.01
