# Sleep-EDF follow-up: same-lr comparison, order and memory controls

Pre-registered: docs/MODI_PROJECT_LOG.md 2026-11-27. Code versions: stage 1 {'ff583b0bc'}, main {'ff583b0bc'}, follow-up {'43afbb75e'}. Test macro-F1 %, folds 0-4; differences in %p, paired by fold.

| model | lr | macro-F1 | accuracy | N1 F1 | syops energy µJ | firing rate |
|---|---|---|---|---|---|---|
| spkall_lif | 0.001 | 73.06 ± 3.65 | 79.75 | 36.6 | 16.142 | 0.4039 |
| spkall_lif | 0.003 | 74.03 ± 3.27 | 81.06 | 34.8 | 16.117 | 0.4055 |
| q_pop_lif | 0.001 | 74.62 ± 3.88 | 81.06 | 36.9 | 18.189 | 0.4492 |
| q_pop_lif | 0.003 | 74.16 ± 3.26 | 80.50 | 37.4 | 18.547 | 0.4531 |
| q_pop_flif | 0.001 | 73.13 ± 3.03 | 79.93 | 34.7 | 18.223 | 0.4486 |
| q_pop_flif | 0.003 | 73.35 ± 3.29 | 79.96 | 36.5 | 18.630 | 0.4528 |
| q_pop_lif_cshuf | 0.003 | 73.63 ± 3.15 | 80.40 | 36.2 | 18.539 | 0.4533 |
| q_pop_lif_nml | 0.003 | 73.41 ± 2.74 | 80.17 | 36.0 | 16.225 | 0.4214 |
| q_pop_flif_cshuf | 0.003 | 73.99 ± 3.19 | 80.74 | 37.0 | 18.448 | 0.4518 |
| q_pop_flif_nml | 0.003 | 72.89 ± 4.06 | 79.96 | 34.6 | 16.492 | 0.4394 |

## L: query form − Spikformer(DE) at the same lr

| neuron | lr | d per fold | mean | positive | verdict |
|---|---|---|---|---|---|
| pop_lif | 0.003 | +0.26 / -3.34 / +1.04 / +2.21 / +0.47 | +0.13 | 4/5 | **no clear difference** |
| pop_lif | 0.001 | +0.24 / +4.44 / +1.07 / +0.77 / +1.26 | +1.56 | 5/5 | **better** |
| pop_flif | 0.003 | -1.42 / -2.49 / +1.07 / +0.80 / -1.34 | -0.68 | 2/5 | **no clear difference** |
| pop_flif | 0.001 | -2.65 / +0.26 / +1.08 / +0.57 / +1.07 | +0.07 | 4/5 | **no clear difference** |

Reading per neuron: pop_lif: **depends on lr**; pop_flif: **not supported**

## O: order — query form − same model with the context epochs shuffled

| neuron | d per fold | mean | positive | verdict |
|---|---|---|---|---|
| pop_lif | +0.69 / -1.05 / +0.63 / +1.52 / +0.87 | +0.53 | 4/5 | **no clear difference** |
| pop_flif | -1.17 / -1.86 / +1.27 / -0.03 / -1.37 | -0.63 | 1/5 | **no clear difference** |

## M: memory — query form − same model with a memoryless Neocortex

| neuron | d per fold | mean | positive | verdict |
|---|---|---|---|---|
| pop_lif | +1.56 / -2.04 / +1.70 / +2.36 / +0.18 | +0.75 | 4/5 | **no clear difference** |
| pop_flif | -0.77 / +1.58 / -1.13 / +1.00 / +1.64 | +0.47 | 3/5 | **no clear difference** |
