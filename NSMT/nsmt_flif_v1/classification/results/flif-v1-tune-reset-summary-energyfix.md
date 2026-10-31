# Tuning stage `reset` (ETT ETTh2 x H96 / H720, seeds 42 / 7 / 2026) — T_h-averaged head energy recounted

See the module docstring for sources, selection and rules. Validation = best validation L1 (selection); test = test MSE of the best-validation weights. Default value marked *.

## validation L1: H96 / H720 cell means over seeds

| setting | spikede* | subtract | zero |
|---|---|---|---|
| mixall_lif_bidir | 0.3159 / 0.5379 | 0.3144 / 0.5375 | 0.3138 / 0.5362 |
| mixall_flif_bidir | 0.3150 / 0.5380 | 0.3148 / 0.5381 | 0.3140 / 0.5379 |
| mixall_pop_lif_bidir | 0.3217 / 0.5457 | n/a | 0.3222 / 0.5458 |
| mixall_pop_flif_bidir | 0.3182 / 0.5436 | n/a | 0.3170 / 0.5423 |
| spkneo_lif | 0.3121 / 0.5374 | 0.3124 / 0.5369 | 0.3115 / 0.5364 |
| spkneo_flif | 0.3117 / 0.5378 | 0.3116 / 0.5358 | 0.3116 / 0.5367 |
| spkneo_pop_lif | 0.3115 / 0.5354 | n/a | 0.3110 / 0.5357 |
| spkneo_pop_flif | 0.3122 / 0.5349 | n/a | 0.3120 / 0.5367 |
| nsmt4_lif | 0.3119 / 0.5359 | 0.3116 / 0.5364 | 0.3107 / 0.5353 |
| nsmt4_flif | 0.3117 / 0.5366 | 0.3122 / 0.5361 | 0.3115 / 0.5356 |
| nsmt4_pop_lif | 0.3108 / 0.5366 | n/a | 0.3102 / 0.5368 |
| nsmt4_pop_flif | 0.3112 / 0.5359 | n/a | 0.3116 / 0.5361 |
| nsmt4q_lif | 0.3119 / 0.5361 | 0.3112 / 0.5346 | 0.3112 / 0.5348 |
| nsmt4q_flif | 0.3110 / 0.5367 | 0.3116 / 0.5361 | 0.3112 / 0.5353 |
| nsmt4q_pop_lif | 0.3123 / 0.5367 | n/a | 0.3120 / 0.5364 |
| nsmt4q_pop_flif | 0.3117 / 0.5368 | n/a | 0.3116 / 0.5366 |

## test MSE: H96 / H720 cell means over seeds

| setting | spikede* | subtract | zero |
|---|---|---|---|
| mixall_lif_bidir | 0.2963 / 0.4293 | 0.2929 / 0.4216 | 0.2907 / 0.4192 |
| mixall_flif_bidir | 0.2931 / 0.4206 | 0.2915 / 0.4185 | 0.2903 / 0.4174 |
| mixall_pop_lif_bidir | 0.3049 / 0.4348 | n/a | 0.3019 / 0.4338 |
| mixall_pop_flif_bidir | 0.2949 / 0.4239 | n/a | 0.2960 / 0.4227 |
| spkneo_lif | 0.2903 / 0.4167 | 0.2881 / 0.4147 | 0.2906 / 0.4159 |
| spkneo_flif | 0.2868 / 0.4175 | 0.2871 / 0.4209 | 0.2911 / 0.4180 |
| spkneo_pop_lif | 0.2941 / 0.4192 | n/a | 0.2931 / 0.4195 |
| spkneo_pop_flif | 0.2897 / 0.4193 | n/a | 0.2929 / 0.4158 |
| nsmt4_lif | 0.2911 / 0.4210 | 0.2912 / 0.4201 | 0.2932 / 0.4172 |
| nsmt4_flif | 0.2910 / 0.4168 | 0.2901 / 0.4168 | 0.2873 / 0.4141 |
| nsmt4_pop_lif | 0.2910 / 0.4101 | n/a | 0.2911 / 0.4093 |
| nsmt4_pop_flif | 0.2915 / 0.4170 | n/a | 0.2923 / 0.4146 |
| nsmt4q_lif | 0.2925 / 0.4177 | 0.2897 / 0.4199 | 0.2909 / 0.4199 |
| nsmt4q_flif | 0.2930 / 0.4165 | 0.2915 / 0.4191 | 0.2899 / 0.4170 |
| nsmt4q_pop_lif | 0.2912 / 0.4135 | n/a | 0.2931 / 0.4159 |
| nsmt4q_pop_flif | 0.2890 / 0.4135 | n/a | 0.2895 / 0.4162 |

## Selection (validation) and verdict 1 (selected − default, test MSE)

| setting | relative validation per value | selected | R selected − default | pairs | verdict 1 | energy sel/default |
|---|---|---|---|---|---|---|
| mixall_lif_bidir | spikede: 1.0000 / subtract: 0.9974 / zero: 0.9951 | zero | -2.10 | 6/6 | better | 0.99 |
| mixall_flif_bidir | spikede: 1.0000 / subtract: 0.9996 / zero: 0.9983 | zero | -0.85 | 6/6 | no clear difference | 0.99 |
| mixall_pop_lif_bidir | spikede: 1.0000 / zero: 1.0008 | spikede | (default) | | default selected | |
| mixall_pop_flif_bidir | spikede: 1.0000 / zero: 0.9970 | zero | +0.06 | 3/6 | no clear difference | 1.00 |
| spkneo_lif | spikede: 1.0000 / subtract: 1.0000 / zero: 0.9983 | zero | -0.01 | 2/6 | no clear difference | 1.00 |
| spkneo_flif | spikede: 1.0000 / subtract: 0.9981 / zero: 0.9989 | subtract | +0.47 | 2/6 | no clear difference | 1.00 |
| spkneo_pop_lif | spikede: 1.0000 / zero: 0.9995 | zero | -0.12 | 3/6 | no clear difference | 1.00 |
| spkneo_pop_flif | spikede: 1.0000 / zero: 1.0013 | spikede | (default) | | default selected | |
| nsmt4_lif | spikede: 1.0000 / subtract: 1.0000 / zero: 0.9975 | zero | -0.08 | 2/6 | no clear difference | 1.00 |
| nsmt4_flif | spikede: 1.0000 / subtract: 1.0004 / zero: 0.9987 | zero | -0.92 | 4/6 | no clear difference | 0.99 |
| nsmt4_pop_lif | spikede: 1.0000 / zero: 0.9992 | zero | -0.09 | 4/6 | no clear difference | 1.00 |
| nsmt4_pop_flif | spikede: 1.0000 / zero: 1.0007 | spikede | (default) | | default selected | |
| nsmt4q_lif | spikede: 1.0000 / subtract: 0.9975 / zero: 0.9976 | subtract | -0.22 | 2/6 | no clear difference | 1.00 |
| nsmt4q_flif | spikede: 1.0000 / subtract: 1.0004 / zero: 0.9989 | zero | -0.48 | 4/6 | no clear difference | 1.00 |
| nsmt4q_pop_lif | spikede: 1.0000 / zero: 0.9993 | zero | +0.61 | 1/6 | no clear difference | 1.00 |
| nsmt4q_pop_flif | spikede: 1.0000 / zero: 0.9997 | zero | +0.40 | 2/6 | no clear difference | 1.00 |
| (report) overall, LIF / f-LIF settings | spikede: 1.0000 (n=8) / subtract: 0.9992 (n=8) / zero: 0.9979 (n=8) | | | | | |
| (report) overall, population settings | spikede: 1.0000 (n=8) / zero: 0.9997 (n=8) | | | | | |

## Verdict 2: the comparisons at the defaults and at each setting's selected value

| comparison | R / verdict at defaults | criterion (2) | R / verdict at selected values | criterion (2) | changed |
|---|---|---|---|---|---|
| mixall_lif_bidir vs spk_ssa | +1.85 (1/6) worse, E 0.18 | not met | -0.30 (4/6) no clear difference, E 0.18 | met | **yes** |
| mixall_flif_bidir vs spk_ssa | +0.27 (2/6) no clear difference, E 0.25 | met | -0.59 (5/6) no clear difference, E 0.25 | met | no |
| mixall_pop_lif_bidir vs spk_ssa | +3.98 (0/6) worse, E 0.46 | not met | +3.98 (0/6) worse, E 0.46 | not met | no |
| mixall_pop_flif_bidir vs spk_ssa | +0.97 (2/6) no clear difference, E 0.97 | not met | +1.02 (1/6) worse, E 0.97 | not met | **yes** |
| spkneo_lif vs spk_ssa | -0.66 (4/6) no clear difference, E 1.30 | not met | -0.70 (3/6) no clear difference, E 1.29 | not met | no |
| spkneo_flif vs spk_ssa | -1.17 (5/6) better, E 1.33 | met | -0.72 (5/6) no clear difference, E 1.33 | not met | **yes** |
| spkneo_pop_lif vs spk_ssa | +0.27 (3/6) no clear difference, E 1.42 | not met | +0.14 (3/6) no clear difference, E 1.42 | not met | no |
| spkneo_pop_flif vs spk_ssa | -0.47 (5/6) no clear difference, E 1.64 | not met | -0.47 (5/6) no clear difference, E 1.64 | not met | no |
| nsmt4_lif vs spk_ssa | -0.02 (3/6) no clear difference, E 1.07 | not met | -0.12 (3/6) no clear difference, E 1.06 | not met | no |
| nsmt4_flif vs spk_ssa | -0.54 (4/6) no clear difference, E 1.10 | not met | -1.49 (5/6) better, E 1.10 | met | **yes** |
| nsmt4_pop_lif vs spk_ssa | -1.33 (5/6) better, E 1.19 | met | -1.42 (4/6) better, E 1.19 | met | no |
| nsmt4_pop_flif vs spk_ssa | -0.42 (5/6) no clear difference, E 1.42 | not met | -0.42 (5/6) no clear difference, E 1.42 | not met | no |
| nsmt4q_lif vs spk_ssa | -0.17 (3/6) no clear difference, E 1.08 | not met | -0.40 (3/6) no clear difference, E 1.08 | not met | no |
| nsmt4q_flif vs spk_ssa | -0.22 (4/6) no clear difference, E 1.11 | not met | -0.70 (5/6) no clear difference, E 1.11 | not met | no |
| nsmt4q_pop_lif vs spk_ssa | -0.89 (4/6) no clear difference, E 1.20 | not met | -0.28 (4/6) no clear difference, E 1.21 | not met | no |
| nsmt4q_pop_flif vs spk_ssa | -1.26 (4/6) better, E 1.43 | met | -0.87 (5/6) no clear difference, E 1.43 | not met | **yes** |
| R1 replace vs append, lif | +0.66 (1/6) no clear difference, E 0.82 |  | +0.59 (1/6) no clear difference, E 0.82 |  | no |
| R1 replace vs append, flif | +0.65 (1/6) no clear difference, E 0.83 |  | -0.76 (5/6) no clear difference, E 0.83 |  | no |
| R1 replace vs append, pop_lif | -1.58 (5/6) better, E 0.83 |  | -1.55 (6/6) better, E 0.84 |  | no |
| R1 replace vs append, pop_flif | +0.05 (2/6) no clear difference, E 0.86 |  | +0.05 (2/6) no clear difference, E 0.86 |  | no |
| Q1 query vs replace, lif | -0.13 (4/6) no clear difference, E 1.01 |  | -0.27 (3/6) no clear difference, E 1.02 |  | no |
| Q1 query vs replace, flif | +0.33 (2/6) no clear difference, E 1.01 |  | +0.82 (2/6) no clear difference, E 1.01 |  | no |
| Q1 query vs replace, pop_lif | +0.45 (1/6) no clear difference, E 1.01 |  | +1.16 (2/6) worse, E 1.01 |  | **yes** |
| Q1 query vs replace, pop_flif | -0.83 (4/6) no clear difference, E 1.01 |  | -0.44 (5/6) no clear difference, E 1.01 |  | no |
| Q2 query vs append, lif | +0.53 (3/6) no clear difference, E 0.83 |  | +0.31 (2/6) no clear difference, E 0.84 |  | no |
| Q2 query vs append, flif | +0.98 (2/6) no clear difference, E 0.84 |  | +0.02 (4/6) no clear difference, E 0.84 |  | no |
| Q2 query vs append, pop_lif | -1.13 (4/6) better, E 0.85 |  | -0.41 (2/6) no clear difference, E 0.85 |  | **yes** |
| Q2 query vs append, pop_flif | -0.79 (4/6) no clear difference, E 0.87 |  | -0.41 (5/6) no clear difference, E 0.87 |  | no |
| W4 f-LIF vs LIF, Neo-mixer every layer | -1.54 (5/6) better, E 1.42 |  | -0.29 (5/6) no clear difference, E 1.43 |  | **yes** |

**Positive (CinC extension rule):** True (any verdict 1 better: True; any verdict 2 changed: True)
