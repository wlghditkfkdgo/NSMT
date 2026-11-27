# Memoryless control: muxlin IAND T4 with every token-axis mixer neuron reset after each token vs normal LIF (one block; ETTh1 / ETTm1 x H96 / H720 x 3 seeds)

Rules: summarize_memoff.py docstring. 12 new runs, commit `4d7ee850b,79b3d99f6`.

## 4-cell means (seed means)

| arm | MSE | MAE | best val L1 | energy µJ (syops) | energy µJ (integer) |
|---|--:|--:|--:|--:|--:|
| muxlin IAND memoryless | 0.4127 | 0.4072 | 0.6110 | 10.35 | 3.37 |
| muxlin IAND (ref, LIF memory) | 0.4117 | 0.4067 | 0.6106 | 10.38 | 3.39 |

## Comparisons (test)

| kind | A | B | MSE R % | lower / higher | MSE verdict | MAE R % | MAE verdict | energy A/B syops | integer |
|---|---|---|--:|---|---|--:|---|--:|--:|
| memory off vs on | muxlin IAND memoryless | muxlin IAND (ref, LIF memory) | +0.28 | 6 / 6 | **no clear difference** | +0.16 | no clear difference | 0.995 | 0.988 |

## Per cell (seed-mean test MSE / MAE, syops energy µJ)

| arm | ETTh1 H96 | ETTh1 H720 | ETTm1 H96 | ETTm1 H720 |
|---|---|---|---|---|
| muxlin IAND memoryless | 0.3815 / 0.3895 / 6.05 | 0.4848 / 0.4607 / 21.47 | 0.3213 / 0.3484 / 3.09 | 0.4634 / 0.4302 / 10.80 |
| muxlin IAND (ref, LIF memory) | 0.3833 / 0.3907 / 6.04 | 0.4788 / 0.4590 / 21.46 | 0.3180 / 0.3448 / 3.15 | 0.4668 / 0.4324 / 10.86 |

Per-cell r (MSE, %) muxlin IAND memoryless vs muxlin IAND (ref, LIF memory): -0.46 +1.26 +1.05 -0.71
