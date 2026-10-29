# B1 residual selection (LIF, fold 0): mean best-validation score over 3 fusions x 3 seeds

| residual | runs | mean validation score | mean test score | mean total µJ (criterion rule) |
|---|---:|---:|---:|---:|
| IAND | 9 | 69.21 | 65.09 | 129.9 |
| OR | 9 | 69.90 | 64.07 | 118.8 |
| ADD | 9 | 69.43 | 63.84 | 120.1 |

**Selected residual: OR** (rule: highest mean best-validation score; ties -> IAND).

## Per condition (test score, best-validation weights; mean ± sd over seeds)

| condition | n | validation | test score | macro-F1 | total µJ syops rule | total µJ integer rule | time s |
|---|---:|---:|---:|---:|---:|---:|---:|
| spk_lif_or_iand | 3 | 67.22 ± 0.84 | 63.08 ± 0.41 | 56.82 ± 1.42 | 119.9 | 119.9 | 5692 |
| spk_lif_or_or | 3 | 67.41 ± 0.56 | 61.78 ± 0.74 | 57.18 ± 0.57 | 110.5 | 110.5 | 5245 |
| spk_lif_or_add | 3 | 66.53 ± 0.65 | 60.36 ± 1.94 | 57.17 ± 1.34 | 414.7 | 111.1 | 6025 |
| spk_lif_sdor_iand | 3 | 69.42 ± 0.36 | 65.07 ± 0.81 | 60.21 ± 1.10 | 126.1 | 126.1 | 5987 |
| spk_lif_sdor_or | 3 | 70.28 ± 0.60 | 63.47 ± 1.13 | 59.77 ± 1.24 | 115.9 | 115.9 | 5969 |
| spk_lif_sdor_add | 3 | 70.41 ± 1.50 | 65.26 ± 0.76 | 60.98 ± 0.83 | 420.2 | 117.7 | 5327 |
| spk_lif_sdlin_iand | 3 | 70.98 ± 0.45 | 67.11 ± 0.86 | 61.38 ± 1.09 | 143.8 | 143.8 | 6867 |
| spk_lif_sdlin_or | 3 | 72.02 ± 2.10 | 66.96 ± 0.95 | 61.74 ± 1.65 | 129.8 | 129.8 | 6060 |
| spk_lif_sdlin_add | 3 | 71.34 ± 0.57 | 65.91 ± 0.27 | 62.01 ± 1.03 | 437.1 | 131.5 | 5147 |
