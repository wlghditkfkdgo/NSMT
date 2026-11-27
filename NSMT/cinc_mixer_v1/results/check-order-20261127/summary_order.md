| model | shuffle | MSE R % (cells ETTh1 96 / 720, ETTm1 96 / 720) | runs higher | range | verdict | MAE R % | MAE verdict |
|---|---|---|--:|---|---|--:|---|
| base | mixer | +4.98 (+2.13 / +4.11 / +6.21 / +7.48) | 12 / 12 | +1.53 .. +10.75 | **order used** | +2.12 | order used |
| base | head | +101.19 (+102.75 / +64.60 / +158.02 / +79.40) | 12 / 12 | +63.15 .. +159.83 | **order used** | +47.87 | order used |
| base | full | +105.54 (+104.36 / +66.83 / +165.16 / +85.80) | 12 / 12 | +63.93 .. +166.22 | **order used** | +50.11 | order used |
| pmux | mixer | +3.85 (+1.12 / +2.00 / +6.25 / +6.03) | 12 / 12 | +0.68 .. +9.13 | **order used** | +1.81 | order used |
| pmux | head | +101.62 (+104.73 / +63.61 / +157.93 / +80.19) | 12 / 12 | +62.26 .. +159.81 | **order used** | +48.29 | order used |
| pmux | full | +105.60 (+105.75 / +65.26 / +165.79 / +85.58) | 12 / 12 | +62.96 .. +168.44 | **order used** | +50.46 | order used |
