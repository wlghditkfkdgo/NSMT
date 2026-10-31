# Tuning stage `layers` (ETT ETTh2 x H96 / H720, seeds 42 / 7 / 2026) — T_h-averaged head energy recounted

See the module docstring for sources, selection and rules. Validation = best validation L1 (selection); test = test MSE of the best-validation weights. Default value marked *.

## validation L1: H96 / H720 cell means over seeds

| setting | 1* | 2 | 3 | 4 |
|---|---|---|---|---|
| mixall_lif_bidir | 0.3159 / 0.5379 | 0.3163 / 0.5388 | 0.3170 / 0.5389 | 0.3160 / 0.5389 |
| mixall_flif_bidir | 0.3150 / 0.5380 | 0.3159 / 0.5388 | 0.3157 / 0.5389 | 0.3164 / 0.5396 |
| mixall_pop_lif_bidir | 0.3217 / 0.5457 | 0.3225 / 0.5484 | 0.3240 / 0.5437 | 0.3236 / 0.5473 |
| mixall_pop_flif_bidir | 0.3182 / 0.5436 | 0.3189 / 0.5434 | 0.3200 / 0.5430 | 0.3201 / 0.5445 |

## test MSE: H96 / H720 cell means over seeds

| setting | 1* | 2 | 3 | 4 |
|---|---|---|---|---|
| mixall_lif_bidir | 0.2963 / 0.4293 | 0.3000 / 0.4273 | 0.2967 / 0.4260 | 0.2971 / 0.4265 |
| mixall_flif_bidir | 0.2931 / 0.4206 | 0.2943 / 0.4200 | 0.2921 / 0.4218 | 0.2896 / 0.4191 |
| mixall_pop_lif_bidir | 0.3049 / 0.4348 | 0.3048 / 0.4349 | 0.3042 / 0.4367 | 0.3048 / 0.4403 |
| mixall_pop_flif_bidir | 0.2949 / 0.4239 | 0.2950 / 0.4229 | 0.2972 / 0.4267 | 0.3019 / 0.4305 |

## Selection (validation) and verdict 1 (selected − default, test MSE)

| setting | relative validation per value | selected | R selected − default | pairs | verdict 1 | energy sel/default |
|---|---|---|---|---|---|---|
| mixall_lif_bidir | 1: 1.0000 / 2: 1.0016 / 3: 1.0026 / 4: 1.0010 | 1 | (default) | | default selected | |
| mixall_flif_bidir | 1: 1.0000 / 2: 1.0021 / 3: 1.0020 / 4: 1.0036 | 1 | (default) | | default selected | |
| mixall_pop_lif_bidir | 1: 1.0000 / 2: 1.0036 / 3: 1.0017 / 4: 1.0044 | 1 | (default) | | default selected | |
| mixall_pop_flif_bidir | 1: 1.0000 / 2: 1.0009 / 3: 1.0022 / 4: 1.0039 | 1 | (default) | | default selected | |

## Verdict 2: the comparisons at the defaults and at each setting's selected value

| comparison | R / verdict at defaults | criterion (2) | R / verdict at selected values | criterion (2) | changed |
|---|---|---|---|---|---|
| mixall_lif_bidir vs spk_ssa | +1.85 (1/6) worse, E 0.18 | not met | +1.85 (1/6) worse, E 0.18 | not met | no |
| mixall_flif_bidir vs spk_ssa | +0.27 (2/6) no clear difference, E 0.25 | met | +0.27 (2/6) no clear difference, E 0.25 | met | no |
| mixall_pop_lif_bidir vs spk_ssa | +3.98 (0/6) worse, E 0.46 | not met | +3.98 (0/6) worse, E 0.46 | not met | no |
| mixall_pop_flif_bidir vs spk_ssa | +0.97 (2/6) no clear difference, E 0.97 | not met | +0.97 (2/6) no clear difference, E 0.97 | not met | no |
| W4 f-LIF vs LIF, Neo-mixer every layer | -1.54 (5/6) better, E 1.42 |  | -1.54 (5/6) better, E 1.42 |  | no |

**Positive (CinC extension rule):** False (any verdict 1 better: False; any verdict 2 changed: False)
