# Sleep-EDF SNN comparison — flif-v1-sleep (+ fold 0 from flif-v1-sleep-lr), sha ff583b0bc

Pre-registered: docs/MODI_PROJECT_LOG.md 2026-11-26. Test macro-F1 %, mean ± std over folds 0-4; differences in %p, paired by fold.

| model | lr | macro-F1 | accuracy | F1 W / N1 / N2 / N3 / REM | syn energy µJ | × Spikformer(DE) | firing rate |
|---|---|---|---|---|---|---|---|
| Spikformer(DE) | 0.001 | 73.06 ± 3.65 | 79.75 | 83.6 / 36.6 / 85.2 / 86.4 / 73.5 | 16.142 | 1.00 | 0.4039 |
| query form, pop LIF | 0.003 | 74.16 ± 3.26 | 80.50 | 85.6 / 37.4 / 85.3 / 87.2 / 75.4 | 18.547 | 1.15 | 0.4531 |
| query form, pop f-LIF | 0.003 | 73.35 ± 3.29 | 79.96 | 83.8 / 36.5 / 85.3 / 87.4 / 73.8 | 18.630 | 1.15 | 0.4528 |
| key/value form, pop LIF | 0.003 | 72.33 ± 4.90 | 78.81 | 83.3 / 35.5 / 84.6 / 86.4 / 71.9 | 16.799 | 1.04 | 0.4460 |
| key/value form, pop f-LIF | 0.003 | 71.92 ± 3.04 | 79.50 | 82.3 / 32.6 / 85.8 / 87.2 / 71.7 | 17.346 | 1.07 | 0.4479 |

## Q1: Neocortex model − Spikformer(DE)

| model | d per fold | mean | positive | verdict |
|---|---|---|---|---|
| query form, pop LIF | +1.68 / -1.30 / +1.23 / +1.20 / +2.66 | +1.10 | 4/5 | **better** |
| query form, pop f-LIF | -0.00 / -0.44 / +1.26 / -0.21 / +0.85 | +0.29 | 2/5 | **no clear difference** |
| key/value form, pop LIF | +0.86 / -2.06 / +1.73 / -1.05 / -3.12 | -0.73 | 2/5 | **no clear difference** |
| key/value form, pop f-LIF | -1.65 / -1.15 / -3.79 / -0.64 / +1.54 | -1.14 | 1/5 | **worse** |

## Q2: query form − key/value form

| neuron | d per fold | mean | positive | verdict |
|---|---|---|---|---|
| pop lif | +0.82 / +0.76 / -0.49 / +2.25 / +5.78 | +1.83 | 4/5 | **query better** |
| pop flif | +1.65 / +0.71 / +5.05 / +0.44 / -0.69 | +1.43 | 4/5 | **query better** |

**Structure (pre-registered rule): query form (1 neuron(s) better than Spikformer(DE), mean Q1 d +0.69 %p).**

Stage-1 validation macro-F1 (fold 0): Spikformer(DE) 0.7795 / 0.7726 -> 0.001; query form, pop LIF 0.7751 / 0.7941 -> 0.003; query form, pop f-LIF 0.7687 / 0.7760 -> 0.003; key/value form, pop LIF 0.7692 / 0.7754 -> 0.003; key/value form, pop f-LIF 0.7675 / 0.7821 -> 0.003
