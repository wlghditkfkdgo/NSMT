# ETT tuning of the fusion MUX — stage lr

Validation-only selection per condition and dataset; test MSE for the verdicts. r = 100 (MSE_A − MSE_B) / MSE_B per seed.

| condition | dataset | candidate | score (val ratio) | val L1 H96 / H720 | test MSE H96 / H720 | selected |
|---|---|---|---:|---|---|---|
| muxlin | ETTh2 | 0.0001 | 1.0004 | 0.3152 / 0.5386 | 0.2921 / 0.4235 |  |
| muxlin | ETTh2 | 0.00025 | 1.0011 | 0.3148 / 0.5400 | 0.2984 / 0.4226 |  |
| muxlin | ETTh2 | 0.0005 (default) | 1.0000 | 0.3145 / 0.5395 | 0.2930 / 0.4214 | **yes** |
| muxlin | ETTh2 | 0.001 | 1.0057 | 0.3186 / 0.5385 | 0.2939 / 0.4216 |  |
| muxlin | ETTh2 | 0.002 | 1.0079 | 0.3187 / 0.5406 | 0.3049 / 0.4268 |  |
| muxlin | ETTm2 | 0.0001 | 0.9943 | 0.2392 / 0.3630 | 0.1793 / 0.4050 | **yes** |
| muxlin | ETTm2 | 0.00025 | 0.9952 | 0.2393 / 0.3635 | 0.1805 / 0.4060 |  |
| muxlin | ETTm2 | 0.0005 (default) | 1.0000 | 0.2402 / 0.3657 | 0.1826 / 0.4107 |  |
| muxlin | ETTm2 | 0.001 | 1.0034 | 0.2408 / 0.3672 | 0.1817 / 0.4135 |  |
| muxlin | ETTm2 | 0.002 | 1.0017 | 0.2402 / 0.3669 | 0.1795 / 0.4107 |  |
| sdlin | ETTh2 | 0.0001 | 0.9993 | 0.3170 / 0.5388 | 0.2937 / 0.4232 |  |
| sdlin | ETTh2 | 0.00025 | 0.9986 | 0.3162 / 0.5395 | 0.2958 / 0.4217 | **yes** |
| sdlin | ETTh2 | 0.0005 (default) | 1.0000 | 0.3173 / 0.5391 | 0.2913 / 0.4265 |  |
| sdlin | ETTh2 | 0.001 | 1.0016 | 0.3194 / 0.5372 | 0.2980 / 0.4339 |  |
| sdlin | ETTh2 | 0.002 | 1.0069 | 0.3216 / 0.5391 | 0.3029 / 0.4282 |  |
| sdlin | ETTm2 | 0.0001 | 0.9926 | 0.2399 / 0.3638 | 0.1805 / 0.4069 | **yes** |
| sdlin | ETTm2 | 0.00025 | 0.9943 | 0.2403 / 0.3644 | 0.1814 / 0.4079 |  |
| sdlin | ETTm2 | 0.0005 (default) | 1.0000 | 0.2415 / 0.3668 | 0.1834 / 0.4133 |  |
| sdlin | ETTm2 | 0.001 | 1.0008 | 0.2410 / 0.3681 | 0.1825 / 0.4147 |  |
| sdlin | ETTm2 | 0.002 | 1.0030 | 0.2419 / 0.3684 | 0.1808 / 0.4159 |  |
| spk_ssa | ETTh2 | 0.0001 | 1.0029 | 0.3118 / 0.5377 | 0.2842 / 0.4186 |  |
| spk_ssa | ETTh2 | 0.00025 | 1.0023 | 0.3122 / 0.5365 | 0.2869 / 0.4166 |  |
| spk_ssa | ETTh2 | 0.0005 (default) | 1.0000 | 0.3115 / 0.5352 | 0.2903 / 0.4223 | **yes** |
| spk_ssa | ETTh2 | 0.001 | 1.0024 | 0.3125 / 0.5360 | 0.2887 / 0.4176 |  |
| spk_ssa | ETTh2 | 0.002 | 1.0056 | 0.3144 / 0.5362 | 0.2943 / 0.4230 |  |
| spk_ssa | ETTm2 | 0.0001 | 0.9976 | 0.2362 / 0.3606 | 0.1768 / 0.4043 |  |
| spk_ssa | ETTm2 | 0.00025 | 0.9975 | 0.2362 / 0.3606 | 0.1767 / 0.4044 | **yes** |
| spk_ssa | ETTm2 | 0.0005 (default) | 1.0000 | 0.2368 / 0.3615 | 0.1775 / 0.4034 |  |
| spk_ssa | ETTm2 | 0.001 | 0.9997 | 0.2366 / 0.3616 | 0.1769 / 0.4049 |  |
| spk_ssa | ETTm2 | 0.002 | 1.0043 | 0.2376 / 0.3634 | 0.1781 / 0.4069 |  |

## Verdict 1: selected vs default (test MSE, per dataset, 6 pairs)

| condition | dataset | selected | R | pairs lower | verdict |
|---|---|---|---:|---|---|
| muxlin | ETTh2 | 0.0005 | +0.00 | 0/6 | no clear difference |
| muxlin | ETTm2 | 0.0001 | -1.61 | 6/6 | better |
| sdlin | ETTh2 | 0.00025 | +0.26 | 1/6 | no clear difference |
| sdlin | ETTm2 | 0.0001 | -1.55 | 6/6 | better |
| spk_ssa | ETTh2 | 0.0005 | +0.00 | 0/6 | no clear difference |
| spk_ssa | ETTm2 | 0.00025 | -0.09 | 5/6 | no clear difference |

**Selection written:** selected_lr.json = {"spkmx_lif_muxlin_or": {"ETTh2": 0.0005, "ETTm2": 0.0001}, "spkmx_lif_sdlin_or": {"ETTh2": 0.00025, "ETTm2": 0.0001}, "spk_ssa": {"ETTh2": 0.0005, "ETTm2": 0.00025}}
