# Tuning stage `alpha` (ETT ETTh2 x H96 / H720, seeds 42 / 7 / 2026)

See the module docstring for sources, selection and rules. Validation = best validation L1 (selection); test = test MSE of the best-validation weights. Default value marked *.

## validation L1: H96 / H720 cell means over seeds

| setting | 0.3 | 0.5* | 0.7 | 0.9 |
|---|---|---|---|---|
| mixall_flif_bidir | 0.3144 / 0.5382 | 0.3140 / 0.5379 | 0.3141 / 0.5381 | 0.3134 / 0.5378 |
| mixall_pop_flif_bidir | 0.3162 / 0.5423 | 0.3170 / 0.5423 | 0.3177 / 0.5439 | 0.3200 / 0.5438 |
| spkneo_flif | 0.3116 / 0.5372 | 0.3116 / 0.5358 | 0.3114 / 0.5362 | 0.3121 / 0.5369 |
| spkneo_pop_flif | 0.3121 / 0.5374 | 0.3122 / 0.5349 | 0.3118 / 0.5372 | 0.3118 / 0.5366 |
| nsmt4_flif | 0.3119 / 0.5352 | 0.3115 / 0.5356 | 0.3116 / 0.5345 | 0.3117 / 0.5359 |
| nsmt4_pop_flif | 0.3109 / 0.5376 | 0.3112 / 0.5359 | 0.3114 / 0.5359 | 0.3107 / 0.5371 |
| nsmt4q_flif | 0.3109 / 0.5360 | 0.3112 / 0.5353 | 0.3114 / 0.5362 | 0.3115 / 0.5353 |
| nsmt4q_pop_flif | 0.3107 / 0.5367 | 0.3116 / 0.5366 | 0.3124 / 0.5368 | 0.3126 / 0.5369 |

## test MSE: H96 / H720 cell means over seeds

| setting | 0.3 | 0.5* | 0.7 | 0.9 |
|---|---|---|---|---|
| mixall_flif_bidir | 0.2900 / 0.4173 | 0.2903 / 0.4174 | 0.2901 / 0.4174 | 0.2912 / 0.4182 |
| mixall_pop_flif_bidir | 0.2927 / 0.4241 | 0.2960 / 0.4227 | 0.3019 / 0.4278 | 0.2993 / 0.4331 |
| spkneo_flif | 0.2917 / 0.4164 | 0.2871 / 0.4209 | 0.2912 / 0.4206 | 0.2920 / 0.4176 |
| spkneo_pop_flif | 0.2904 / 0.4230 | 0.2897 / 0.4193 | 0.2927 / 0.4206 | 0.2895 / 0.4196 |
| nsmt4_flif | 0.2889 / 0.4180 | 0.2873 / 0.4141 | 0.2897 / 0.4174 | 0.2913 / 0.4151 |
| nsmt4_pop_flif | 0.2911 / 0.4116 | 0.2915 / 0.4170 | 0.2932 / 0.4131 | 0.2924 / 0.4143 |
| nsmt4q_flif | 0.2891 / 0.4150 | 0.2899 / 0.4170 | 0.2897 / 0.4151 | 0.2886 / 0.4178 |
| nsmt4q_pop_flif | 0.2884 / 0.4157 | 0.2895 / 0.4162 | 0.2898 / 0.4186 | 0.2905 / 0.4151 |

## Selection (validation) and verdict 1 (selected − default, test MSE)

| setting | relative validation per value | selected | R selected − default | pairs | verdict 1 | energy sel/default |
|---|---|---|---|---|---|---|
| mixall_flif_bidir | 0.3: 1.0009 / 0.5: 1.0000 / 0.7: 1.0003 / 0.9: 0.9989 | 0.9 | +0.25 | 2/6 | no clear difference | 1.00 |
| mixall_pop_flif_bidir | 0.3: 0.9987 / 0.5: 1.0000 / 0.7: 1.0025 / 0.9: 1.0062 | 0.3 | -0.39 | 3/6 | no clear difference | 0.99 |
| spkneo_flif | 0.3: 1.0013 / 0.5: 1.0000 / 0.7: 0.9999 / 0.9: 1.0017 | 0.7 | +0.67 | 2/6 | no clear difference | 1.00 |
| spkneo_pop_flif | 0.3: 1.0022 / 0.5: 1.0000 / 0.7: 1.0013 / 0.9: 1.0008 | 0.5 | (default) | | default selected | |
| nsmt4_flif | 0.3: 1.0003 / 0.5: 1.0000 / 0.7: 0.9992 / 0.9: 1.0006 | 0.7 | +0.84 | 2/6 | no clear difference | 1.00 |
| nsmt4_pop_flif | 0.3: 1.0010 / 0.5: 1.0000 / 0.7: 1.0002 / 0.9: 1.0003 | 0.5 | (default) | | default selected | |
| nsmt4q_flif | 0.3: 1.0002 / 0.5: 1.0000 / 0.7: 1.0011 / 0.9: 1.0004 | 0.5 | (default) | | default selected | |
| nsmt4q_pop_flif | 0.3: 0.9986 / 0.5: 1.0000 / 0.7: 1.0014 / 0.9: 1.0018 | 0.3 | -0.24 | 4/6 | no clear difference | 1.00 |

## Verdict 2: the comparisons at the defaults and at each setting's selected value

| comparison | R / verdict at defaults | criterion (2) | R / verdict at selected values | criterion (2) | changed |
|---|---|---|---|---|---|
| mixall_flif_bidir vs spk_ssa | -0.59 (5/6) no clear difference, E 0.12 | met | -0.34 (5/6) no clear difference, E 0.11 | met | no |
| mixall_pop_flif_bidir vs spk_ssa | +1.02 (1/6) worse, E 0.46 | not met | +0.62 (1/6) no clear difference, E 0.45 | met | **yes** |
| spkneo_flif vs spk_ssa | -0.72 (5/6) no clear difference, E 1.16 | not met | -0.06 (3/6) no clear difference, E 1.16 | not met | no |
| spkneo_pop_flif vs spk_ssa | -0.47 (5/6) no clear difference, E 1.30 | not met | -0.47 (5/6) no clear difference, E 1.30 | not met | no |
| nsmt4_flif vs spk_ssa | -1.49 (5/6) better, E 1.05 | met | -0.68 (5/6) no clear difference, E 1.04 | not met | **yes** |
| nsmt4_pop_flif vs spk_ssa | -0.42 (5/6) no clear difference, E 1.20 | not met | -0.42 (5/6) no clear difference, E 1.20 | not met | no |
| nsmt4q_flif vs spk_ssa | -0.70 (5/6) no clear difference, E 1.05 | not met | -0.70 (5/6) no clear difference, E 1.05 | not met | no |
| nsmt4q_pop_flif vs spk_ssa | -0.87 (5/6) no clear difference, E 1.20 | not met | -1.11 (6/6) better, E 1.20 | met | **yes** |
| mixall_flif_bidir vs mixall_lif_bidir | -0.29 (5/6) no clear difference, E 1.43 |  | -0.03 (2/6) no clear difference, E 1.42 |  | no |
| mixall_pop_flif_bidir vs mixall_pop_lif_bidir | -2.84 (6/6) better, E 2.12 |  | -3.22 (6/6) better, E 2.10 |  | no |
| spkneo_flif vs spkneo_lif | -0.01 (3/6) no clear difference, E 1.01 |  | +0.65 (2/6) no clear difference, E 1.01 |  | no |
| spkneo_pop_flif vs spkneo_pop_lif | -0.60 (4/6) no clear difference, E 1.09 |  | -0.60 (4/6) no clear difference, E 1.09 |  | no |
| nsmt4_flif vs nsmt4_lif | -1.36 (5/6) better, E 1.02 |  | -0.55 (3/6) no clear difference, E 1.01 |  | **yes** |
| nsmt4_pop_flif vs nsmt4_pop_lif | +1.02 (1/6) worse, E 1.10 |  | +1.02 (1/6) worse, E 1.10 |  | no |
| nsmt4q_flif vs nsmt4q_lif | -0.28 (3/6) no clear difference, E 1.01 |  | -0.28 (3/6) no clear difference, E 1.01 |  | no |
| nsmt4q_pop_flif vs nsmt4q_pop_lif | -0.57 (4/6) no clear difference, E 1.10 |  | -0.80 (5/6) no clear difference, E 1.10 |  | no |

**Positive (CinC extension rule):** True (any verdict 1 better: False; any verdict 2 changed: True)
