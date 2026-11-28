# SF-de (Spikformer-style, SSA + relative position bias) on ETTh1 / ETTm1: order shuffle and accuracy vs the base muxlin IAND

| model | shuffle | MSE R % (ETTh1 96 / 720, ETTm1 96 / 720) | runs higher | verdict |
|---|---|---|--:|---|
| muxlin IAND (base) | mixer | +4.98 (+2.13 / +4.11 / +6.21 / +7.48) | 12 / 12 | **order used** |
| muxlin IAND (base) | head | +101.19 (+102.75 / +64.60 / +158.02 / +79.40) | 12 / 12 | **order used** |
| muxlin IAND (base) | full | +105.54 (+104.36 / +66.83 / +165.16 / +85.80) | 12 / 12 | **order used** |
| SF-de | SSA | +0.49 (+0.46 / +1.07 / +0.13 / +0.32) | 12 / 12 | **no evidence** |
| SF-de | head | +103.29 (+105.53 / +64.31 / +161.22 / +82.09) | 12 / 12 | **order used** |
| SF-de | full | +103.70 (+105.79 / +64.70 / +161.72 / +82.59) | 12 / 12 | **order used** |

Paired contrast d_base(mixer) - d_SF-de(SSA): mean +4.49 %p, larger 12 / smaller 0 -> **muxlin depends more**

## Accuracy (report only): SF-de vs muxlin IAND

- MSE: R -0.36 % (cells -1.81 / +1.85 / -0.34 / -1.13), lower 8 / higher 4 -> no clear difference
- MAE: R +0.20 % (cells -0.25 / +1.52 / +0.21 / -0.68), lower 6 / higher 6 -> no clear difference
- 4-cell means: SF-de MSE 0.4106 / MAE 0.4077 / 19.79 uJ; muxlin IAND 0.4117 / 0.4067 / 10.38 uJ; energy ratio SF-de / muxlin 1.895
