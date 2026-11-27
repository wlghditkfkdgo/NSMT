| model | shuffle | MSE R % (cells ETTh2 96 / 720, ETTm2 96 / 720) | runs higher | range | verdict | MAE R % |
|---|---|---|--:|---|---|--:|
| muxlin (first chain, pt16) | mixer | +0.73 (-0.27 / +0.82 / +2.13 / +0.22) | 10 / 12 | -0.88 .. +2.42 | **no evidence** | +0.60 |
| muxlin (first chain, pt16) | head | +21.72 (+28.41 / +10.00 / +37.61 / +10.86) | 12 / 12 | +9.77 .. +37.79 | **order used** | +15.10 |
| muxlin (first chain, pt16) | full | +21.75 (+28.05 / +10.07 / +37.67 / +11.19) | 12 / 12 | +9.82 .. +37.81 | **order used** | +15.14 |
| SF-de (SSA + rel. bias, pt16) | SSA | +0.15 (-0.01 / +0.07 / +0.56 / -0.02) | 8 / 12 | -0.11 .. +0.63 | **no evidence** | +0.17 |
| SF-de (SSA + rel. bias, pt16) | head | +22.45 (+30.34 / +9.89 / +38.12 / +11.43) | 12 / 12 | +9.53 .. +38.49 | **order used** | +15.46 |
| SF-de (SSA + rel. bias, pt16) | full | +22.36 (+30.36 / +9.91 / +37.73 / +11.45) | 12 / 12 | +9.56 .. +37.99 | **order used** | +15.46 |

Paired contrast d_muxlin(mixer) - d_SF-de(SSA): mean +0.58 %p, larger 10 / smaller 2 of 12 -> **no difference**
