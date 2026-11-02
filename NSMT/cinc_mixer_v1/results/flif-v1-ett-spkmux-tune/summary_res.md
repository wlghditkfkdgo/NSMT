# ETT tuning of the fusion MUX — stage res

Validation-only selection per condition and dataset; test MSE for the verdicts. r = 100 (MSE_A − MSE_B) / MSE_B per seed.

| condition | dataset | candidate | score (val ratio) | val L1 H96 / H720 | test MSE H96 / H720 | selected |
|---|---|---|---:|---|---|---|
| muxlin | ETTh2 | or (default) | 1.0000 | 0.3145 / 0.5395 | 0.2930 / 0.4214 |  |
| muxlin | ETTh2 | iand | 0.9938 | 0.3126 / 0.5359 | 0.2952 / 0.4218 | **yes** |
| muxlin | ETTh2 | add | 1.0041 | 0.3180 / 0.5379 | 0.2892 / 0.4296 |  |
| muxlin | ETTm2 | or (default) | 1.0000 | 0.2387 / 0.3627 | 0.1796 / 0.4047 |  |
| muxlin | ETTm2 | iand | 0.9927 | 0.2365 / 0.3608 | 0.1775 / 0.4034 | **yes** |
| muxlin | ETTm2 | add | 0.9998 | 0.2384 / 0.3629 | 0.1797 / 0.4061 |  |
| sdlin | ETTh2 | or (default) | 1.0000 | 0.3162 / 0.5395 | 0.2958 / 0.4217 |  |
| sdlin | ETTh2 | iand | 0.9924 | 0.3126 / 0.5374 | 0.2890 / 0.4205 | **yes** |
| sdlin | ETTh2 | add | 1.0056 | 0.3187 / 0.5412 | 0.2943 / 0.4243 |  |
| sdlin | ETTm2 | or (default) | 1.0000 | 0.2392 / 0.3635 | 0.1798 / 0.4053 |  |
| sdlin | ETTm2 | iand | 0.9912 | 0.2365 / 0.3611 | 0.1775 / 0.4044 | **yes** |
| sdlin | ETTm2 | add | 1.0019 | 0.2400 / 0.3636 | 0.1814 / 0.4051 |  |

## Verdict 1: selected vs default (test MSE, per dataset, 6 pairs)

| condition | dataset | selected | R | pairs lower | verdict |
|---|---|---|---:|---|---|
| muxlin | ETTh2 | iand | +0.43 | 2/6 | no clear difference |
| muxlin | ETTm2 | iand | -0.73 | 6/6 | no clear difference |
| sdlin | ETTh2 | iand | -1.29 | 5/6 | better |
| sdlin | ETTm2 | iand | -0.75 | 4/6 | no clear difference |

**Selection written:** selected_res.json = {"spkmx_lif_muxlin_or": {"ETTh2": "iand", "ETTm2": "iand"}, "spkmx_lif_sdlin_or": {"ETTh2": "iand", "ETTm2": "iand"}}
