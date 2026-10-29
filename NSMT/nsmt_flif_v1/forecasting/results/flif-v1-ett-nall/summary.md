# Whole-model neuron swap `flif-v1-ett-nall` vs neocortex-only swap `flif-v1-gen-ett` — summary

Test MSE, best-validation weights; mean (seed 42 / 7 / 2026). R = relative MSE difference per seed. Energy = total_fft_uJ.

## ETTh2, horizon 96

| neuron | MSE, neocortex only | MSE, every layer | R every − neocortex only | verdict | R every − Spikformer-style | energy every / neocortex only | energy every / Spikformer-style |
|---|---|---|---|---|---|---|---|
| LIF | 0.2913 (0.2916 / 0.2931 / 0.2891) | 0.2963 (0.2987 / 0.2949 / 0.2953) | +1.73 % (+2.41 / +0.64 / +2.13) | worse | +2.05 % (+2.42 / +1.61 / +2.13) | 1.01 | 0.12 |
| f-LIF | 0.2921 (0.2900 / 0.2909 / 0.2954) | 0.2926 (0.2941 / 0.2928 / 0.2908) | +0.17 % (+1.43 / +0.65 / -1.58) | no clear difference | +0.77 % (+0.87 / +0.85 / +0.57) | 1.21 | 0.17 |
| population LIF | 0.2932 (0.2948 / 0.2922 / 0.2927) | 0.3048 (0.3069 / 0.3021 / 0.3053) | +3.94 % (+4.11 / +3.40 / +4.31) | worse | +4.98 % (+5.25 / +4.08 / +5.61) | 1.55 | 0.33 |
| population f-LIF | 0.2912 (0.2898 / 0.2914 / 0.2924) | 0.2949 (0.2910 / 0.2975 / 0.2961) | +1.27 % (+0.43 / +2.11 / +1.28) | worse | +1.57 % (-0.21 / +2.50 / +2.43) | 1.85 | 0.70 |

Spikformer-style MSE 0.2903 (0.2916 / 0.2903 / 0.2891), energy 13.78 (13.83 / 13.73 / 13.77) µJ

