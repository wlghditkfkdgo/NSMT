# Tuning stage `lr` (ETT ETTh2 x H96 / H720, seeds 42 / 7 / 2026) — T_h-averaged head energy recounted

See the module docstring for sources, selection and rules. Validation = best validation L1 (selection); test = test MSE of the best-validation weights. Default value marked *.

## validation L1: H96 / H720 cell means over seeds

| setting | 0.00025 | 0.0005* | 0.001 | 0.002 |
|---|---|---|---|---|
| spk_ssa | 0.3122 / 0.5365 | 0.3115 / 0.5352 | 0.3125 / 0.5360 | 0.3144 / 0.5362 |
| mixall_lif_bidir | 0.3160 / 0.5386 | 0.3159 / 0.5379 | 0.3168 / 0.5387 | 0.3197 / 0.5392 |
| mixall_flif_bidir | 0.3150 / 0.5380 | 0.3153 / 0.5384 | 0.3166 / 0.5390 | 0.3191 / 0.5403 |
| mixall_pop_lif_bidir | 0.3253 / 0.5529 | 0.3245 / 0.5500 | 0.3230 / 0.5471 | 0.3217 / 0.5457 |
| mixall_pop_flif_bidir | 0.3186 / 0.5437 | 0.3182 / 0.5436 | 0.3178 / 0.5444 | 0.3192 / 0.5446 |
| spkneo_lif | 0.3128 / 0.5367 | 0.3121 / 0.5374 | 0.3135 / 0.5374 | 0.3150 / 0.5370 |
| spkneo_flif | 0.3117 / 0.5378 | 0.3117 / 0.5378 | 0.3140 / 0.5363 | 0.3146 / 0.5373 |
| spkneo_pop_lif | 0.3117 / 0.5359 | 0.3115 / 0.5354 | 0.3121 / 0.5362 | 0.3148 / 0.5370 |
| spkneo_pop_flif | 0.3128 / 0.5354 | 0.3122 / 0.5349 | 0.3134 / 0.5359 | 0.3152 / 0.5379 |
| nsmt4_lif | 0.3123 / 0.5374 | 0.3119 / 0.5359 | 0.3123 / 0.5360 | 0.3155 / 0.5368 |
| nsmt4_flif | 0.3122 / 0.5363 | 0.3117 / 0.5366 | 0.3133 / 0.5359 | 0.3147 / 0.5381 |
| nsmt4_pop_lif | 0.3110 / 0.5363 | 0.3108 / 0.5366 | 0.3116 / 0.5363 | 0.3141 / 0.5368 |
| nsmt4_pop_flif | 0.3118 / 0.5366 | 0.3112 / 0.5359 | 0.3122 / 0.5367 | 0.3139 / 0.5361 |
| nsmt4q_lif | 0.3124 / 0.5366 | 0.3119 / 0.5361 | 0.3133 / 0.5368 | 0.3144 / 0.5378 |
| nsmt4q_flif | 0.3121 / 0.5367 | 0.3110 / 0.5367 | 0.3122 / 0.5369 | 0.3144 / 0.5362 |
| nsmt4q_pop_lif | 0.3129 / 0.5363 | 0.3123 / 0.5367 | 0.3133 / 0.5365 | 0.3156 / 0.5363 |
| nsmt4q_pop_flif | 0.3125 / 0.5374 | 0.3117 / 0.5368 | 0.3131 / 0.5378 | 0.3153 / 0.5379 |

## test MSE: H96 / H720 cell means over seeds

| setting | 0.00025 | 0.0005* | 0.001 | 0.002 |
|---|---|---|---|---|
| spk_ssa | 0.2869 / 0.4166 | 0.2903 / 0.4223 | 0.2887 / 0.4176 | 0.2943 / 0.4230 |
| mixall_lif_bidir | 0.2982 / 0.4239 | 0.2963 / 0.4293 | 0.2959 / 0.4305 | 0.2975 / 0.4264 |
| mixall_flif_bidir | 0.2931 / 0.4206 | 0.2926 / 0.4238 | 0.2953 / 0.4216 | 0.2945 / 0.4229 |
| mixall_pop_lif_bidir | 0.3047 / 0.4303 | 0.3048 / 0.4341 | 0.3020 / 0.4312 | 0.3049 / 0.4348 |
| mixall_pop_flif_bidir | 0.2975 / 0.4255 | 0.2949 / 0.4239 | 0.2988 / 0.4286 | 0.2991 / 0.4291 |
| spkneo_lif | 0.2861 / 0.4310 | 0.2903 / 0.4167 | 0.2899 / 0.4160 | 0.2918 / 0.4221 |
| spkneo_flif | 0.2882 / 0.4189 | 0.2868 / 0.4175 | 0.2872 / 0.4216 | 0.2883 / 0.4211 |
| spkneo_pop_lif | 0.2897 / 0.4154 | 0.2941 / 0.4192 | 0.2947 / 0.4164 | 0.2894 / 0.4205 |
| spkneo_pop_flif | 0.2859 / 0.4200 | 0.2897 / 0.4193 | 0.2908 / 0.4145 | 0.2922 / 0.4211 |
| nsmt4_lif | 0.2894 / 0.4198 | 0.2911 / 0.4210 | 0.2925 / 0.4180 | 0.2974 / 0.4216 |
| nsmt4_flif | 0.2890 / 0.4200 | 0.2910 / 0.4168 | 0.2901 / 0.4192 | 0.2908 / 0.4168 |
| nsmt4_pop_lif | 0.2893 / 0.4130 | 0.2910 / 0.4101 | 0.2945 / 0.4142 | 0.2973 / 0.4173 |
| nsmt4_pop_flif | 0.2910 / 0.4143 | 0.2915 / 0.4170 | 0.2929 / 0.4150 | 0.3050 / 0.4182 |
| nsmt4q_lif | 0.2875 / 0.4169 | 0.2925 / 0.4177 | 0.2888 / 0.4199 | 0.2926 / 0.4239 |
| nsmt4q_flif | 0.2878 / 0.4178 | 0.2930 / 0.4165 | 0.2938 / 0.4164 | 0.2970 / 0.4249 |
| nsmt4q_pop_lif | 0.2883 / 0.4160 | 0.2912 / 0.4135 | 0.2925 / 0.4179 | 0.2987 / 0.4221 |
| nsmt4q_pop_flif | 0.2842 / 0.4157 | 0.2890 / 0.4135 | 0.2911 / 0.4138 | 0.2887 / 0.4209 |

## Selection (validation) and verdict 1 (selected − default, test MSE)

| setting | relative validation per value | selected | R selected − default | pairs | verdict 1 | energy sel/default |
|---|---|---|---|---|---|---|
| spk_ssa | 0.00025: 1.0023 / 0.0005: 1.0000 / 0.001: 1.0024 / 0.002: 1.0056 | 0.0005 | (default) | | default selected | |
| mixall_lif_bidir | 0.00025: 1.0008 / 0.0005: 1.0000 / 0.001: 1.0022 / 0.002: 1.0072 | 0.0005 | (default) | | default selected | |
| mixall_flif_bidir | 0.00025: 0.9992 / 0.0005: 1.0000 / 0.001: 1.0026 / 0.002: 1.0079 | 0.00025 | -0.28 | 3/6 | no clear difference | 1.00 |
| mixall_pop_lif_bidir | 0.00025: 1.0039 / 0.0005: 1.0000 / 0.001: 0.9949 / 0.002: 0.9917 | 0.002 | +0.11 | 4/6 | no clear difference | 0.99 |
| mixall_pop_flif_bidir | 0.00025: 1.0008 / 0.0005: 1.0000 / 0.001: 1.0002 / 0.002: 1.0025 | 0.0005 | (default) | | default selected | |
| spkneo_lif | 0.00025: 1.0006 / 0.0005: 1.0000 / 0.001: 1.0023 / 0.002: 1.0044 | 0.0005 | (default) | | default selected | |
| spkneo_flif | 0.00025: 1.0001 / 0.0005: 1.0000 / 0.001: 1.0023 / 0.002: 1.0042 | 0.0005 | (default) | | default selected | |
| spkneo_pop_lif | 0.00025: 1.0009 / 0.0005: 1.0000 / 0.001: 1.0017 / 0.002: 1.0067 | 0.0005 | (default) | | default selected | |
| spkneo_pop_flif | 0.00025: 1.0013 / 0.0005: 1.0000 / 0.001: 1.0028 / 0.002: 1.0075 | 0.0005 | (default) | | default selected | |
| nsmt4_lif | 0.00025: 1.0020 / 0.0005: 1.0000 / 0.001: 1.0007 / 0.002: 1.0066 | 0.0005 | (default) | | default selected | |
| nsmt4_flif | 0.00025: 1.0006 / 0.0005: 1.0000 / 0.001: 1.0018 / 0.002: 1.0062 | 0.0005 | (default) | | default selected | |
| nsmt4_pop_lif | 0.00025: 1.0001 / 0.0005: 1.0000 / 0.001: 1.0010 / 0.002: 1.0055 | 0.0005 | (default) | | default selected | |
| nsmt4_pop_flif | 0.00025: 1.0015 / 0.0005: 1.0000 / 0.001: 1.0022 / 0.002: 1.0044 | 0.0005 | (default) | | default selected | |
| nsmt4q_lif | 0.00025: 1.0012 / 0.0005: 1.0000 / 0.001: 1.0029 / 0.002: 1.0056 | 0.0005 | (default) | | default selected | |
| nsmt4q_flif | 0.00025: 1.0017 / 0.0005: 1.0000 / 0.001: 1.0020 / 0.002: 1.0049 | 0.0005 | (default) | | default selected | |
| nsmt4q_pop_lif | 0.00025: 1.0006 / 0.0005: 1.0000 / 0.001: 1.0014 / 0.002: 1.0049 | 0.0005 | (default) | | default selected | |
| nsmt4q_pop_flif | 0.00025: 1.0018 / 0.0005: 1.0000 / 0.001: 1.0031 / 0.002: 1.0068 | 0.0005 | (default) | | default selected | |

## Verdict 2: the comparisons at the defaults and at each setting's selected value

| comparison | R / verdict at defaults | criterion (2) | R / verdict at selected values | criterion (2) | changed |
|---|---|---|---|---|---|
| mixall_lif_bidir vs spk_ssa | +1.85 (1/6) worse, E 0.18 | not met | +1.85 (1/6) worse, E 0.18 | not met | no |
| mixall_flif_bidir vs spk_ssa | +0.55 (1/6) no clear difference, E 0.25 | met | +0.27 (2/6) no clear difference, E 0.25 | met | no |
| mixall_pop_lif_bidir vs spk_ssa | +3.89 (0/6) worse, E 0.46 | not met | +3.98 (0/6) worse, E 0.46 | not met | no |
| mixall_pop_flif_bidir vs spk_ssa | +0.97 (2/6) no clear difference, E 0.97 | not met | +0.97 (2/6) no clear difference, E 0.97 | not met | no |
| spkneo_lif vs spk_ssa | -0.66 (4/6) no clear difference, E 1.30 | not met | -0.66 (4/6) no clear difference, E 1.30 | not met | no |
| spkneo_flif vs spk_ssa | -1.17 (5/6) better, E 1.33 | met | -1.17 (5/6) better, E 1.33 | met | no |
| spkneo_pop_lif vs spk_ssa | +0.27 (3/6) no clear difference, E 1.42 | not met | +0.27 (3/6) no clear difference, E 1.42 | not met | no |
| spkneo_pop_flif vs spk_ssa | -0.47 (5/6) no clear difference, E 1.64 | not met | -0.47 (5/6) no clear difference, E 1.64 | not met | no |
| nsmt4_lif vs spk_ssa | -0.02 (3/6) no clear difference, E 1.07 | not met | -0.02 (3/6) no clear difference, E 1.07 | not met | no |
| nsmt4_flif vs spk_ssa | -0.54 (4/6) no clear difference, E 1.10 | not met | -0.54 (4/6) no clear difference, E 1.10 | not met | no |
| nsmt4_pop_lif vs spk_ssa | -1.33 (5/6) better, E 1.19 | met | -1.33 (5/6) better, E 1.19 | met | no |
| nsmt4_pop_flif vs spk_ssa | -0.42 (5/6) no clear difference, E 1.42 | not met | -0.42 (5/6) no clear difference, E 1.42 | not met | no |
| nsmt4q_lif vs spk_ssa | -0.17 (3/6) no clear difference, E 1.08 | not met | -0.17 (3/6) no clear difference, E 1.08 | not met | no |
| nsmt4q_flif vs spk_ssa | -0.22 (4/6) no clear difference, E 1.11 | not met | -0.22 (4/6) no clear difference, E 1.11 | not met | no |
| nsmt4q_pop_lif vs spk_ssa | -0.89 (4/6) no clear difference, E 1.20 | not met | -0.89 (4/6) no clear difference, E 1.20 | not met | no |
| nsmt4q_pop_flif vs spk_ssa | -1.26 (4/6) better, E 1.43 | met | -1.26 (4/6) better, E 1.43 | met | no |
| R1 replace vs append, lif | +0.66 (1/6) no clear difference, E 0.82 |  | +0.66 (1/6) no clear difference, E 0.82 |  | no |
| R1 replace vs append, flif | +0.65 (1/6) no clear difference, E 0.83 |  | +0.65 (1/6) no clear difference, E 0.83 |  | no |
| R1 replace vs append, pop_lif | -1.58 (5/6) better, E 0.83 |  | -1.58 (5/6) better, E 0.83 |  | no |
| R1 replace vs append, pop_flif | +0.05 (2/6) no clear difference, E 0.86 |  | +0.05 (2/6) no clear difference, E 0.86 |  | no |
| Q1 query vs replace, lif | -0.13 (4/6) no clear difference, E 1.01 |  | -0.13 (4/6) no clear difference, E 1.01 |  | no |
| Q1 query vs replace, flif | +0.33 (2/6) no clear difference, E 1.01 |  | +0.33 (2/6) no clear difference, E 1.01 |  | no |
| Q1 query vs replace, pop_lif | +0.45 (1/6) no clear difference, E 1.01 |  | +0.45 (1/6) no clear difference, E 1.01 |  | no |
| Q1 query vs replace, pop_flif | -0.83 (4/6) no clear difference, E 1.01 |  | -0.83 (4/6) no clear difference, E 1.01 |  | no |
| Q2 query vs append, lif | +0.53 (3/6) no clear difference, E 0.83 |  | +0.53 (3/6) no clear difference, E 0.83 |  | no |
| Q2 query vs append, flif | +0.98 (2/6) no clear difference, E 0.84 |  | +0.98 (2/6) no clear difference, E 0.84 |  | no |
| Q2 query vs append, pop_lif | -1.13 (4/6) better, E 0.85 |  | -1.13 (4/6) better, E 0.85 |  | no |
| Q2 query vs append, pop_flif | -0.79 (4/6) no clear difference, E 0.87 |  | -0.79 (4/6) no clear difference, E 0.87 |  | no |

**Positive (CinC extension rule):** False (any verdict 1 better: False; any verdict 2 changed: False)
