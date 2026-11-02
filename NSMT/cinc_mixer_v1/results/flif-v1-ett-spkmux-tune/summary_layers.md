# ETT tuning of the fusion MUX — stage layers

Validation-only selection per condition and dataset; test MSE for the verdicts. r = 100 (MSE_A − MSE_B) / MSE_B per seed.

| condition | dataset | candidate | score (val ratio) | val L1 H96 / H720 | test MSE H96 / H720 | selected |
|---|---|---|---:|---|---|---|
| muxlin | ETTh2 | 1 (default) | 1.0000 | 0.3145 / 0.5395 | 0.2930 / 0.4214 | **yes** |
| muxlin | ETTh2 | 2 | 1.0022 | 0.3163 / 0.5387 | 0.2888 / 0.4264 |  |
| muxlin | ETTh2 | 3 | 1.0017 | 0.3162 / 0.5383 | 0.2919 / 0.4280 |  |
| muxlin | ETTm2 | 1 | 0.9984 | 0.2387 / 0.3627 | 0.1796 / 0.4047 | **yes** |
| muxlin | ETTm2 | 2 (default) | 1.0000 | 0.2392 / 0.3630 | 0.1793 / 0.4050 |  |
| muxlin | ETTm2 | 3 | 1.0012 | 0.2396 / 0.3633 | 0.1799 / 0.4055 |  |
| sdlin | ETTh2 | 1 (default) | 1.0000 | 0.3162 / 0.5395 | 0.2958 / 0.4217 | **yes** |
| sdlin | ETTh2 | 2 | 1.0015 | 0.3168 / 0.5400 | 0.2978 / 0.4204 |  |
| sdlin | ETTh2 | 3 | 1.0037 | 0.3181 / 0.5402 | 0.2948 / 0.4223 |  |
| sdlin | ETTm2 | 1 | 0.9980 | 0.2392 / 0.3635 | 0.1798 / 0.4053 | **yes** |
| sdlin | ETTm2 | 2 (default) | 1.0000 | 0.2399 / 0.3638 | 0.1805 / 0.4069 |  |
| sdlin | ETTm2 | 3 | 1.0006 | 0.2400 / 0.3641 | 0.1806 / 0.4053 |  |

## Verdict 1: selected vs default (test MSE, per dataset, 6 pairs)

| condition | dataset | selected | R | pairs lower | verdict |
|---|---|---|---:|---|---|
| muxlin | ETTh2 | 1 | +0.00 | 0/6 | no clear difference |
| muxlin | ETTm2 | 1 | +0.05 | 3/6 | no clear difference |
| sdlin | ETTh2 | 1 | +0.00 | 0/6 | no clear difference |
| sdlin | ETTm2 | 1 | -0.41 | 4/6 | no clear difference |

**Selection written:** selected_layers.json = {"spkmx_lif_muxlin_or": {"ETTh2": 1, "ETTm2": 1}, "spkmx_lif_sdlin_or": {"ETTh2": 1, "ETTm2": 1}}
