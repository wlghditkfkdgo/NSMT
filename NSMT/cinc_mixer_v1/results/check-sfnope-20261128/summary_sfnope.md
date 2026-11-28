# SF-de without position information (ssa_pe 0) on MIT-BIH and ETTh1 / ETTm1

Rules: summarize_sfnope.py docstring. New runs: MIT-BIH 3 (commit `7af763e56`), ETT 12 (commit `7af763e56`).

## MIT-BIH (seed means)

| arm | macro-F1 % | accuracy % | class S F1 % | energy µJ (syops) |
|---|--:|--:|--:|--:|
| SF-de PE off (new) | 91.27 | 98.33 | 82.07 | 7.62 |
| SF-de PE on | 91.94 | 98.53 | 85.14 | 7.80 |
| SpkMixer LIF memory | 91.99 | 98.56 | 85.94 | 37.06 |
| SpkMixer memoryless | 90.13 | 98.03 | 78.59 | 36.15 |

| kind | A | B | d %p (42 / 7 / 2026) | mean %p | verdict | energy A/B |
|---|---|---|---|--:|---|--:|
| M1 | SF-de PE off (new) | SF-de PE on | -0.32 / -1.39 / -0.29 | -0.67 | **A worse** | 0.977 |
| M2 (main) | SpkMixer LIF memory | SF-de PE off (new) | +0.45 / +1.02 / +0.67 | +0.71 | **A better** | 4.863 |
| M3 | SpkMixer memoryless | SF-de PE off (new) | -1.26 / -0.76 / -1.40 | -1.14 | **A worse** | 4.744 |
| M4 | SpkMixer LIF memory | SF-de PE on | +0.13 / -0.37 / +0.37 | +0.04 | **no clear difference** | 4.751 |

Order check (trained PE-off SF-de, mean of 3 permutation seeds): SSA shuffle drop +0.0000 / +0.0000 / +0.0000 %p, full shuffle drop +0.0000 / +0.0000 / +0.0000 %p (|drop| < 0.05 required)

## ETT (4-cell means of seed means)

| arm | MSE | MAE | energy µJ (syops) |
|---|--:|--:|--:|
| SF-de PE off (new) | 0.4120 | 0.4081 | 19.82 |
| SF-de PE on | 0.4106 | 0.4077 | 19.79 |
| muxlin IAND LIF memory | 0.4117 | 0.4067 | 10.38 |
| muxlin IAND memoryless | 0.4127 | 0.4072 | 10.35 |

| kind | A | B | MSE R % | lower / higher | MSE verdict | MAE R % | MAE verdict | energy A/B |
|---|---|---|--:|---|---|--:|---|--:|
| E1 | SF-de PE off (new) | SF-de PE on | +0.33 | 4 / 8 | **no clear difference** | +0.13 | no clear difference | 1.001 |
| E2 (main) | muxlin IAND LIF memory | SF-de PE off (new) | +0.06 | 5 / 7 | **no clear difference** | -0.32 | no clear difference | 0.529 |
| E3 | muxlin IAND memoryless | SF-de PE off (new) | +0.32 | 7 / 5 | **no clear difference** | -0.17 | no clear difference | 0.527 |

| arm | ETTh1 H96 | ETTh1 H720 | ETTm1 H96 | ETTm1 H720 |
|---|---|---|---|---|
| SF-de PE off (new) | 0.3778 / 0.3901 | 0.4893 / 0.4655 | 0.3175 / 0.3464 | 0.4633 / 0.4305 |
| SF-de PE on | 0.3764 / 0.3897 | 0.4877 / 0.4660 | 0.3169 / 0.3455 | 0.4614 / 0.4294 |
| muxlin IAND LIF memory | 0.3833 / 0.3907 | 0.4788 / 0.4590 | 0.3180 / 0.3448 | 0.4668 / 0.4324 |
| muxlin IAND memoryless | 0.3815 / 0.3895 | 0.4848 / 0.4607 | 0.3213 / 0.3484 | 0.4634 / 0.4302 |

Order shuffle of the trained PE-off SF-de (test MSE change %, mean over 12 runs; ETTh1 / ETTm1): mixer -0.00 (-0.00 / +0.00, max |.| 0.000); head +102.51 (+84.38 / +120.64, max |.| 160.941); full +102.51 (+84.38 / +120.64, max |.| 160.941) (SSA shuffle |change| < 0.05 % required)
