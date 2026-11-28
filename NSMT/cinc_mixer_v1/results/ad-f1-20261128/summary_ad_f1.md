# AD residual with precision / recall / F1 (post hoc; no training)

Rules and sources: ad_f1.py docstring. Seed means (%); ± = seed std of F1.

## point-adjusted (TSLib / NSMT Table 9 convention)

| arm | MSL P / R / F1 | SMAP P / R / F1 | PSM P / R / F1 | avg F1 |
|---|---|---|---|--:|
| IAND-all | 73.54 / 33.05 / **45.57** ± 2.82 | 89.59 / 54.48 / **67.76** ± 0.77 | 98.01 / 87.82 / **92.63** ± 0.03 | 68.65 |
| G5 (MSL / SMAP OR, PSM IAND) | 70.72 / 29.15 / **41.29** ± 0.24 | 89.72 / 55.19 / **68.35** ± 0.02 | 98.01 / 87.82 / **92.63** ± 0.03 | 67.42 |
| ADD (11-19 validation choice) | 73.57 / 33.28 / **45.79** ± 3.01 | 89.78 / 55.13 / **68.32** ± 0.02 | 98.04 / 89.15 / **93.38** ± 0.09 | 69.16 |
| SF-de(tuned) | 73.55 / 33.41 / **45.92** ± 2.81 | 89.62 / 55.05 / **68.21** ± 0.10 | 97.96 / 87.29 / **92.31** ± 0.94 | 68.81 |
| ETSformer (NSMT Table 9) | 85.13 / 84.93 / 85.03 | 92.25 / 55.75 / 69.50 | 90.02 / 80.36 / 84.91 | 79.81 |
| PatchTST (NSMT Table 9) | 88.34 / 70.96 / 78.70 | 90.64 / 55.46 / 68.82 | 91.10 / 80.94 / 85.72 | 77.75 |
| Informer (NSMT Table 9) | 81.77 / 86.48 / 84.06 | 90.11 / 57.13 / 69.92 | 64.27 / 96.33 / 77.10 | 77.03 |
| Autoformer (NSMT Table 9) | 77.27 / 80.92 / 79.05 | 90.40 / 58.62 / 71.12 | 99.08 / 88.15 / 93.29 | 81.15 |
| FEDformer (NSMT Table 9) | 77.14 / 80.07 / 78.57 | 90.47 / 58.10 / 70.76 | 97.31 / 97.16 / 97.23 | 82.19 |
| NSMT (paper) (NSMT Table 9) | 88.24 / 67.54 / 76.51 | 91.61 / 64.57 / 75.75 | 98.54 / 92.56 / 95.46 | 82.57 |
| training-free random score (11-03) | — / — / 84.60 | — / — / 93.75 | — / — / 95.89 | 91.41 |

## raw (per point, no adjustment)

| arm | MSL P / R / F1 | SMAP P / R / F1 | PSM P / R / F1 | avg F1 |
|---|---|---|---|--:|
| IAND-all | 27.66 / 4.52 / **7.78** ± 0.03 | 18.83 / 1.47 / **2.72** ± 0.10 | 60.20 / 2.70 / **5.17** ± 0.06 | 5.22 |
| G5 (MSL / SMAP OR, PSM IAND) | 26.83 / 4.43 / **7.60** ± 0.05 | 19.92 / 1.57 / **2.92** ± 0.11 | 60.20 / 2.70 / **5.17** ± 0.06 | 5.23 |
| ADD (11-19 validation choice) | 27.26 / 4.46 / **7.66** ± 0.08 | 20.65 / 1.63 / **3.03** ± 0.13 | 60.97 / 2.79 / **5.33** ± 0.09 | 5.34 |
| SF-de(tuned) | 27.22 / 4.47 / **7.68** ± 0.07 | 17.84 / 1.39 / **2.57** ± 0.08 | 60.75 / 2.81 / **5.38** ± 0.01 | 5.21 |
| training-free random score (11-03) | — / — / 1.82 | — / — / 1.86 | — / — / 1.93 | 1.87 |

## Comparisons (AD rule form: R >= +2 % and >= 2/3 pairs; adoption-bar form: mean F1 difference >= 0.001, syops energy <= 1.5x)

| metric | kind | A vs B | datasets | R % | per-dataset r % | pairs higher | verdict | mean Δ | energy A/B | bar |
|---|---|---|---|--:|---|--:|---|--:|--:|---|
| pa_f1 | F1 | IAND-all vs SF-de(tuned) | MSL / SMAP / PSM | -0.35 | -0.75 / -0.66 / +0.36 | 3/9 | no clear difference | -0.0016 | 1.018 | fail |
| pa_f1 | F2 | IAND-all vs G5 (MSL / SMAP OR, PSM IAND) | MSL / SMAP | +4.76 | +10.38 / -0.86 | 3/6 | no clear difference | +0.0185 | 1.032 | pass |
| pa_f1 | F0 | G5 (MSL / SMAP OR, PSM IAND) vs SF-de(tuned) | MSL / SMAP / PSM | -3.06 | -9.74 / +0.20 / +0.36 | 5/9 | no clear difference | -0.0139 | 0.997 | fail |
| pa_f1 | F3 | ADD (11-19 validation choice) vs SF-de(tuned) | MSL / SMAP / PSM | +0.55 | +0.32 / +0.16 / +1.17 | 7/9 | no clear difference | +0.0035 | 1.207 | pass |
| pa_f1 | F4 | IAND-all vs ADD (11-19 validation choice) | MSL / SMAP / PSM | -0.48 | +0.17 / -0.82 / -0.80 | 2/9 | no clear difference | -0.0051 | 0.846 | fail |
| raw_f1 | F1 | IAND-all vs SF-de(tuned) | MSL / SMAP / PSM | +1.20 | +1.26 / +6.15 / -3.79 | 4/9 | no clear difference | +0.0001 | 1.018 | fail |
| raw_f1 | F2 | IAND-all vs G5 (MSL / SMAP OR, PSM IAND) | MSL / SMAP | -2.04 | +2.34 / -6.42 | 3/6 | no clear difference | -0.0001 | 1.032 | fail |
| raw_f1 | F0 | G5 (MSL / SMAP OR, PSM IAND) vs SF-de(tuned) | MSL / SMAP / PSM | +2.88 | -1.06 / +13.49 / -3.79 | 3/9 | no clear difference | +0.0002 | 0.997 | fail |
| raw_f1 | F3 | ADD (11-19 validation choice) vs SF-de(tuned) | MSL / SMAP / PSM | +5.53 | -0.27 / +17.73 / -0.86 | 4/9 | no clear difference | +0.0013 | 1.207 | pass |
| raw_f1 | F4 | IAND-all vs ADD (11-19 validation choice) | MSL / SMAP / PSM | -3.70 | +1.53 / -9.73 / -2.91 | 4/9 | no clear difference | -0.0012 | 0.846 | fail |
| aucpr | F1 | IAND-all vs SF-de(tuned) | MSL / SMAP / PSM | +0.50 | +0.46 / +1.07 / -0.02 | 6/9 | no clear difference | +0.0006 | 1.018 | fail |
| aucpr | F2 | IAND-all vs G5 (MSL / SMAP OR, PSM IAND) | MSL / SMAP | -1.39 | -1.75 / -1.03 | 1/6 | no clear difference | -0.0019 | 1.032 | fail |
| aucpr | F0 | G5 (MSL / SMAP OR, PSM IAND) vs SF-de(tuned) | MSL / SMAP / PSM | +1.45 | +2.25 / +2.12 / -0.02 | 8/9 | no clear difference | +0.0019 | 0.997 | pass |
| aucpr | F3 | ADD (11-19 validation choice) vs SF-de(tuned) | MSL / SMAP / PSM | +0.43 | +0.24 / +2.06 / -1.02 | 5/9 | no clear difference | -0.0004 | 1.207 | fail |
| aucpr | F4 | IAND-all vs ADD (11-19 validation choice) | MSL / SMAP / PSM | +0.09 | +0.22 / -0.97 / +1.02 | 6/9 | no clear difference | +0.0010 | 0.846 | pass |

Same recommendation rule as the AUC-PR test (F2 not worse and F1 passes the bar): point-adjusted F1 -> **keep G5**, raw F1 -> **keep G5**
