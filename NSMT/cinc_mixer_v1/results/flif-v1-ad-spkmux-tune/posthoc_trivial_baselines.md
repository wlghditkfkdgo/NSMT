# Post-hoc reference scores (report only, not pre-registered)

Same test points, threshold rule and metrics as the final runs; no training. Values in %.

| dataset | score | AUC-PR | AUC-ROC | raw F1 | point-adjusted F1 | anomaly share |
|---|---|---:|---:|---:|---:|---:|
| MSL | random | 10.53 | 49.97 | 1.82 | 84.60 | 10.55 |
| MSL | train-dev | 14.02 | 53.27 | 6.43 | 82.06 | 10.55 |
| MSL | win-mean | 14.96 | 64.07 | 4.47 | 81.79 | 10.55 |
| SMAP | random | 12.79 | 49.99 | 1.86 | 93.75 | 12.79 |
| SMAP | train-dev | 10.56 | 41.31 | 1.23 | 67.07 | 12.79 |
| SMAP | win-mean | 10.84 | 43.50 | 1.18 | 67.39 | 12.79 |
| PSM | random | 27.75 | 50.01 | 1.93 | 95.89 | 27.74 |
| PSM | train-dev | 47.26 | 65.81 | 3.30 | 88.32 | 27.74 |
| PSM | win-mean | 38.12 | 58.53 | 4.19 | 92.42 | 27.74 |
