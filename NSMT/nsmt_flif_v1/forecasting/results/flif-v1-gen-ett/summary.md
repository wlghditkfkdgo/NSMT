# Generalization batch, ETT forecasting `flif-v1-gen-ett` — summary

Best-validation weights (early stopping on the validation L1, patience 10) on the test set; MSE / MAE on the scaled data, mean over seeds (42, 7, 2026).

## Test MSE (mean over seeds; n < 3 marked)

| condition | ETTh1 H96 | ETTh1 H192 | ETTh1 H336 | ETTh1 H720 | ETTh2 H96 | ETTh2 H192 | ETTh2 H336 | ETTh2 H720 | ETTm1 H96 | ETTm1 H192 | ETTm1 H336 | ETTm1 H720 | ETTm2 H96 | ETTm2 H192 | ETTm2 H336 | ETTm2 H720 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 0.3782 | 0.4403 | 0.4753 | 0.4960 | 0.2903 | 0.3643 | 0.4102 | 0.4223 | 0.3277 | 0.3893 | 0.4044 | 0.4716 | 0.1775 | 0.2428 | 0.3036 | 0.4034 |
| Neo-mixer charges + |H_f - H_r|, spikeDE LIF | 0.3900 | 0.4367 | 0.4796 | 0.4778 | 0.2945 | 0.3687 | 0.4166 | 0.4209 | 0.3291 | 0.3835 | 0.4102 | 0.4674 | 0.1791 | 0.2436 | 0.3059 | 0.4048 |
| Neo-mixer charges + |H_f - H_r|, f-LIF | 0.3876 | 0.4409 | 0.4867 | 0.4817 | 0.2923 | 0.3664 | 0.4199 | 0.4202 | 0.3340 | 0.3819 | 0.4105 | 0.4671 | 0.1786 | 0.2444 | 0.3045 | 0.4052 |
| Neo-mixer charges + |H_f - H_r|, population LIF (psi 0.5) | 0.3902 | 0.4540 | 0.4920 | 0.4883 | 0.2958 | 0.3675 | 0.4174 | 0.4299 | 0.3320 | 0.4023 | 0.4114 | 0.4688 | 0.1796 | 0.2460 | 0.3090 | 0.4064 |
| Neo-mixer charges + |H_f - H_r|, population f-LIF (psi 1) | 0.3893 | 0.4417 | 0.4894 | 0.4792 | 0.2921 | 0.3658 | 0.4137 | 0.4217 | 0.3318 | 0.3866 | 0.4177 | 0.4703 | 0.1788 | 0.2447 | 0.3083 | 0.4053 |
| Neo-mixer OR, spikeDE LIF | 0.3859 | 0.4442 | 0.4825 | 0.4841 | 0.2913 | 0.3670 | 0.4120 | 0.4199 | 0.3261 | 0.3852 | 0.4050 | 0.4706 | 0.1789 | 0.2439 | 0.3054 | 0.4049 |
| Neo-mixer OR, f-LIF | 0.3879 | 0.4443 | 0.4817 | 0.4786 | 0.2921 | 0.3667 | 0.4179 | 0.4203 | 0.3267 | 0.3799 | 0.4084 | 0.4703 | 0.1785 | 0.2436 | 0.3052 | 0.4050 |
| Neo-mixer OR, population LIF (psi 0.5) | 0.3917 | 0.4468 | 0.4869 | 0.4842 | 0.2932 | 0.3734 | 0.4182 | 0.4206 | 0.3312 | 0.3845 | 0.4116 | 0.4676 | 0.1785 | 0.2454 | 0.3067 | 0.4062 |
| Neo-mixer OR, population f-LIF (psi 1) | 0.3869 | 0.4425 | 0.4902 | 0.4764 | 0.2912 | 0.3729 | 0.4117 | 0.4198 | 0.3249 | 0.3908 | 0.4114 | 0.4640 | 0.1780 | 0.2441 | 0.3058 | 0.4029 |

## Test MAE (mean over seeds; n < 3 marked)

| condition | ETTh1 H96 | ETTh1 H192 | ETTh1 H336 | ETTh1 H720 | ETTh2 H96 | ETTh2 H192 | ETTh2 H336 | ETTh2 H720 | ETTm1 H96 | ETTm1 H192 | ETTm1 H336 | ETTm1 H720 | ETTm2 H96 | ETTm2 H192 | ETTm2 H336 | ETTm2 H720 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 0.3882 | 0.4225 | 0.4411 | 0.4678 | 0.3350 | 0.3825 | 0.4183 | 0.4366 | 0.3524 | 0.3806 | 0.3976 | 0.4378 | 0.2536 | 0.2981 | 0.3378 | 0.3959 |
| Neo-mixer charges + |H_f - H_r|, spikeDE LIF | 0.4000 | 0.4286 | 0.4498 | 0.4652 | 0.3386 | 0.3861 | 0.4240 | 0.4373 | 0.3532 | 0.3801 | 0.4033 | 0.4382 | 0.2564 | 0.2993 | 0.3394 | 0.3966 |
| Neo-mixer charges + |H_f - H_r|, f-LIF | 0.3997 | 0.4292 | 0.4509 | 0.4695 | 0.3370 | 0.3848 | 0.4256 | 0.4364 | 0.3569 | 0.3806 | 0.4024 | 0.4363 | 0.2563 | 0.2999 | 0.3394 | 0.3967 |
| Neo-mixer charges + |H_f - H_r|, population LIF (psi 0.5) | 0.4027 | 0.4357 | 0.4534 | 0.4706 | 0.3402 | 0.3863 | 0.4246 | 0.4412 | 0.3556 | 0.3891 | 0.4017 | 0.4376 | 0.2548 | 0.3000 | 0.3399 | 0.3975 |
| Neo-mixer charges + |H_f - H_r|, population f-LIF (psi 1) | 0.4017 | 0.4319 | 0.4520 | 0.4690 | 0.3376 | 0.3847 | 0.4224 | 0.4372 | 0.3539 | 0.3820 | 0.4061 | 0.4378 | 0.2543 | 0.2992 | 0.3402 | 0.3970 |
| Neo-mixer OR, spikeDE LIF | 0.3973 | 0.4288 | 0.4483 | 0.4668 | 0.3369 | 0.3851 | 0.4225 | 0.4363 | 0.3521 | 0.3811 | 0.4001 | 0.4369 | 0.2560 | 0.2999 | 0.3390 | 0.3963 |
| Neo-mixer OR, f-LIF | 0.3978 | 0.4291 | 0.4484 | 0.4652 | 0.3372 | 0.3851 | 0.4247 | 0.4367 | 0.3534 | 0.3794 | 0.4014 | 0.4367 | 0.2559 | 0.2997 | 0.3391 | 0.3967 |
| Neo-mixer OR, population LIF (psi 0.5) | 0.4007 | 0.4310 | 0.4508 | 0.4672 | 0.3388 | 0.3898 | 0.4253 | 0.4378 | 0.3530 | 0.3809 | 0.4025 | 0.4368 | 0.2542 | 0.2988 | 0.3391 | 0.3975 |
| Neo-mixer OR, population f-LIF (psi 1) | 0.3982 | 0.4294 | 0.4496 | 0.4642 | 0.3370 | 0.3888 | 0.4221 | 0.4368 | 0.3496 | 0.3835 | 0.4008 | 0.4354 | 0.2542 | 0.2989 | 0.3388 | 0.3961 |

## Energy and size (mean over the 16 cells and seeds)

| condition | runs | total µJ (FFT) per window | neocortex firing | epochs run | params (H96 / H720) |
|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 48 | 22.99 | n/a | 14.8 | 80670–174192 / 559902–1132656 |
| Neo-mixer charges + |H_f - H_r|, spikeDE LIF | 48 | 9.93 | 0.0554 ± 0.0125 | 15.9 | 83712–185856 / 562944–1144320 |
| Neo-mixer charges + |H_f - H_r|, f-LIF | 48 | 10.41 | 0.0504 ± 0.0089 | 15.2 | 83712–185856 / 562944–1144320 |
| Neo-mixer charges + |H_f - H_r|, population LIF (psi 0.5) | 48 | 11.86 | 0.0168 ± 0.0073 | 14.8 | 84224–186880 / 563456–1145344 |
| Neo-mixer charges + |H_f - H_r|, population f-LIF (psi 1) | 48 | 15.23 | 0.0214 ± 0.0071 | 14.9 | 84224–186880 / 563456–1145344 |
| Neo-mixer OR, spikeDE LIF | 48 | 2.01 | 0.0600 ± 0.0136 | 16.8 | 80576–173440 / 559808–1131904 |
| Neo-mixer OR, f-LIF | 48 | 2.50 | 0.0540 ± 0.0103 | 17.5 | 80576–173440 / 559808–1131904 |
| Neo-mixer OR, population LIF (psi 0.5) | 48 | 3.97 | 0.0526 ± 0.0261 | 16.4 | 81088–174464 / 560320–1132928 |
| Neo-mixer OR, population f-LIF (psi 1) | 48 | 7.35 | 0.0466 ± 0.0245 | 17.2 | 81088–174464 / 560320–1132928 |

## Pre-registered comparisons (test MSE; r = 100 (A − B) / B per seed, %)

| question | A | B | cells better / none / worse | R = mean of cell means | seed pairs A < B | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|---|
| G1 linmx LIF vs Spikformer (primary, criterion 2) | Neo-mixer charges + |H_f - H_r|, spikeDE LIF | Spikformer-style SSA (T=4) | 2 / 9 / 5 | +0.34 | 16/48 | no clear difference | 0.49 | met |
| G2 linmx vs Spikformer (criterion 2) | Neo-mixer charges + |H_f - H_r|, f-LIF | Spikformer-style SSA (T=4) | 2 / 9 / 5 | +0.51 | 16/48 | no clear difference | 0.52 | not met |
| G2 linmx vs Spikformer (criterion 2) | Neo-mixer charges + |H_f - H_r|, population LIF (psi 0.5) | Spikformer-style SSA (T=4) | 1 / 3 / 12 | +1.59 | 8/48 | worse | 0.59 | not met |
| G2 linmx vs Spikformer (criterion 2) | Neo-mixer charges + |H_f - H_r|, population f-LIF (psi 1) | Spikformer-style SSA (T=4) | 1 / 10 / 5 | +0.74 | 16/48 | no clear difference | 0.77 | not met |
| G3 linmx vs OR (fusion) | Neo-mixer charges + |H_f - H_r|, spikeDE LIF | Neo-mixer OR, spikeDE LIF | 2 / 11 / 3 | +0.10 | 23/48 | no clear difference | 4.84 |  |
| G3 linmx vs OR (fusion) | Neo-mixer charges + |H_f - H_r|, f-LIF | Neo-mixer OR, f-LIF | 0 / 14 / 2 | +0.26 | 18/48 | no clear difference | 4.11 |  |
| G3 linmx vs OR (fusion) | Neo-mixer charges + |H_f - H_r|, population LIF (psi 0.5) | Neo-mixer OR, population LIF (psi 0.5) | 1 / 11 / 4 | +0.71 | 16/48 | no clear difference | 2.98 |  |
| G3 linmx vs OR (fusion) | Neo-mixer charges + |H_f - H_r|, population f-LIF (psi 1) | Neo-mixer OR, population f-LIF (psi 1) | 1 / 12 / 3 | +0.40 | 15/48 | no clear difference | 2.09 |  |
| G4 OR population LIF vs Spikformer (criterion 2) | Neo-mixer OR, population LIF (psi 0.5) | Spikformer-style SSA (T=4) | 1 / 5 / 10 | +0.90 | 9/48 | no clear difference | 0.21 | met |
| G5 (report) OR vs Spikformer (criterion 2) | Neo-mixer OR, spikeDE LIF | Spikformer-style SSA (T=4) | 1 / 13 / 2 | +0.24 | 13/48 | no clear difference | 0.10 | met |
| G5 (report) OR vs Spikformer (criterion 2) | Neo-mixer OR, f-LIF | Spikformer-style SSA (T=4) | 2 / 11 / 3 | +0.25 | 17/48 | no clear difference | 0.13 | met |
| G5 (report) OR vs Spikformer (criterion 2) | Neo-mixer OR, population f-LIF (psi 1) | Spikformer-style SSA (T=4) | 2 / 10 / 4 | +0.35 | 16/48 | no clear difference | 0.38 | met |

## Per-cell mean r (%) and cell verdict (b = better, w = worse, . = no clear difference)

| A vs B | ETTh1 H96 | ETTh1 H192 | ETTh1 H336 | ETTh1 H720 | ETTh2 H96 | ETTh2 H192 | ETTh2 H336 | ETTh2 H720 | ETTm1 H96 | ETTm1 H192 | ETTm1 H336 | ETTm1 H720 | ETTm2 H96 | ETTm2 H192 | ETTm2 H336 | ETTm2 H720 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mixlinmx_lif − spk_ssa | +3.1 w | -0.8 . | +0.9 . | -3.7 b | +1.4 w | +1.2 w | +1.5 w | -0.3 . | +0.5 . | -1.4 b | +1.4 w | -0.9 . | +0.9 . | +0.3 . | +0.8 . | +0.3 . |
| mixlinmx_flif − spk_ssa | +2.5 w | +0.1 . | +2.4 w | -2.9 b | +0.7 . | +0.6 . | +2.4 w | -0.5 . | +2.0 w | -1.8 b | +1.5 w | -0.9 . | +0.7 . | +0.6 . | +0.3 . | +0.4 . |
| mixlinmx_pop_lif − spk_ssa | +3.2 w | +3.1 w | +3.5 w | -1.6 b | +1.9 w | +0.9 . | +1.8 w | +1.8 w | +1.4 w | +3.3 w | +1.7 w | -0.5 . | +1.2 w | +1.3 w | +1.8 w | +0.7 . |
| mixlinmx_pop_flif − spk_ssa | +2.9 w | +0.3 . | +3.0 w | -3.4 b | +0.6 . | +0.4 . | +0.8 . | -0.1 . | +1.3 w | -0.6 . | +3.3 w | -0.2 . | +0.7 . | +0.8 . | +1.6 w | +0.4 . |
| mixlinmx_lif − mix_lif_bidir | +1.1 w | -1.7 b | -0.6 . | -1.3 b | +1.1 w | +0.4 . | +1.1 . | +0.3 . | +0.9 . | -0.4 . | +1.3 w | -0.7 . | +0.1 . | -0.1 . | +0.2 . | -0.0 . |
| mixlinmx_flif − mix_flif_bidir | -0.1 . | -0.8 . | +1.0 w | +0.7 . | +0.1 . | -0.1 . | +0.5 . | -0.0 . | +2.2 w | +0.5 . | +0.5 . | -0.7 . | +0.1 . | +0.3 . | -0.2 . | +0.0 . |
| mixlinmx_pop_lif − mix_pop_lif_bidir | -0.4 . | +1.6 w | +1.1 w | +0.9 . | +0.9 . | -1.5 b | -0.1 . | +2.2 w | +0.3 . | +4.6 w | -0.0 . | +0.3 . | +0.6 . | +0.3 . | +0.8 . | +0.0 . |
| mixlinmx_pop_flif − mix_pop_flif_bidir | +0.6 . | -0.2 . | -0.1 . | +0.7 . | +0.3 . | -1.9 b | +0.5 . | +0.5 . | +2.1 w | -1.0 . | +1.5 w | +1.4 w | +0.5 . | +0.2 . | +0.8 . | +0.6 . |
| mix_pop_lif_bidir − spk_ssa | +3.6 w | +1.5 w | +2.4 w | -2.4 b | +1.0 w | +2.5 w | +1.9 w | -0.4 . | +1.2 w | -1.1 . | +1.8 w | -0.8 . | +0.6 . | +1.0 w | +1.0 w | +0.7 . |
| mix_lif_bidir − spk_ssa | +2.0 w | +0.9 . | +1.5 w | -2.4 b | +0.3 . | +0.8 . | +0.4 . | -0.6 . | -0.4 . | -0.9 . | +0.2 . | -0.2 . | +0.8 . | +0.5 . | +0.6 . | +0.4 . |
| mix_flif_bidir − spk_ssa | +2.6 w | +0.9 . | +1.4 w | -3.5 b | +0.6 . | +0.7 . | +1.9 w | -0.5 . | -0.2 . | -2.3 b | +1.0 . | -0.2 . | +0.6 . | +0.3 . | +0.5 . | +0.4 . |
| mix_pop_flif_bidir − spk_ssa | +2.3 w | +0.5 . | +3.1 w | -4.0 b | +0.3 . | +2.3 w | +0.4 . | -0.6 . | -0.8 . | +0.4 . | +1.8 w | -1.6 b | +0.3 . | +0.5 . | +0.7 . | -0.1 . |
