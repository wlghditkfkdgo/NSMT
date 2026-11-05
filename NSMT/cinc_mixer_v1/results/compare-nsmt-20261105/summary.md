# SpkMixer muxlin vs the NSMT paper and the Spikformer-style model (report only, 2026-11-05)

Sources and definitions: see the module docstring.

## ETT test MSE / MAE (scaled data; seeds 42, 7, 2026, mean; paper = one run)

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean MSE |
|---|---|---|---|---|---|
| muxlin (validation-selected) | 0.2952 / 0.3381 | 0.4218 / 0.4384 | 0.1775 / 0.2547 | 0.4034 / 0.3955 | 0.3245 |
| muxlin (default settings) | 0.2930 / 0.3374 | 0.4214 / 0.4387 | 0.1826 / 0.2605 | 0.4107 / 0.4007 | 0.3269 |
| muxlin (lowest test MSE per cell, post hoc) | 0.2888 / 0.3350 | 0.4214 / 0.4387 | 0.1775 / 0.2547 | 0.4034 / 0.3955 | 0.3228 |
| NSMT rerun-b (same rule as muxlin) | 0.2896 / 0.3345 | 0.4153 / 0.4358 | 0.1744 / 0.2503 | 0.4014 / 0.3931 | 0.3202 |
| NSMT rerun-a (paper rule) | 0.2835 / 0.3312 | 0.4097 / 0.4299 | 0.1750 / 0.2514 | 0.4002 / 0.3932 | 0.3171 |
| NSMT paper (Table 14) | 0.288 / 0.334 | 0.409 / 0.429 | 0.175 / 0.251 | 0.399 / 0.393 | 0.3177 |
| Spikformer-style (tuned) | 0.2903 / 0.3350 | 0.4223 / 0.4366 | 0.1767 / 0.2533 | 0.4044 / 0.3960 | 0.3234 |
| Spikformer-style (default) | 0.2903 / 0.3350 | 0.4223 / 0.4366 | 0.1775 / 0.2536 | 0.4034 / 0.3959 | 0.3234 |
| SpikF (paper Table 14) | 0.290 / 0.336 | 0.416 / 0.436 | 0.175 / 0.251 | 0.405 / 0.397 | 0.3215 |
| iSpikformer (paper Table 14) | 0.308 / 0.349 | 0.438 / 0.439 | 0.178 / 0.256 | 0.406 / 0.398 | 0.3325 |

Post-hoc settings (residual, lr, mixer layers): ETTh2 H96 or, 0.0005, 2; ETTh2 H720 or, 0.0005, 1; ETTm2 H96 iand, 0.0001, 1; ETTm2 H720 iand, 0.0001, 1

## ETT energy (µJ per window; total / synapses only)

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| muxlin (validation-selected) | 9.06 / 5.80 | 10.67 / 7.44 | 9.89 / 6.36 | 11.49 / 7.97 |
| NSMT rerun-b | 56.78 / 33.77 | 68.57 / 45.59 | 64.93 / 38.94 | 71.40 / 46.63 |
| NSMT rerun-a | 56.00 / 33.17 | 66.89 / 44.09 | 63.79 / 38.24 | 71.62 / 46.77 |
| Spikformer-style (tuned) | 9.30 / 5.58 | 10.79 / 7.08 | 10.29 / 6.27 | 11.54 / 7.57 |
| NSMT paper (Table 6, synapses) | — | — / 44.90 | — | — / 46.72 |

## ETT paired comparisons (R % = mean of per-cell seed-mean r; rule: R <= -1 % and >= 8 / 12 lower)

| A vs B | R % | lower pairs | verdict | energy A/B | per cell r % |
|---|---:|---:|---|---:|---|
| muxlin (selected) vs NSMT rerun-b | +1.46 | 0/12 | worse | 0.16 | ETTh2 H96 +1.95; ETTh2 H720 +1.57; ETTm2 H96 +1.81; ETTm2 H720 +0.50 |
| muxlin (selected) vs NSMT rerun-a | +2.33 | 0/12 | worse | 0.16 | ETTh2 H96 +4.15; ETTh2 H720 +2.95; ETTm2 H96 +1.43; ETTm2 H720 +0.81 |
| muxlin (selected) vs Spikformer-style (tuned) | +0.45 | 3/12 | no clear difference | 0.98 | ETTh2 H96 +1.67; ETTh2 H720 -0.12; ETTm2 H96 +0.47; ETTm2 H720 -0.23 |
| NSMT rerun-b vs Spikformer-style (tuned) | -0.99 | 11/12 | no clear difference | 6.24 | ETTh2 H96 -0.27; ETTh2 H720 -1.66; ETTm2 H96 -1.31; ETTm2 H720 -0.73 |

## MIT-BIH test (70 / 30 random beat split; seeds 42, 7, 2026)

| model | macro-F1 % | accuracy % | F1 N / S / V / F / Q % | energy total µJ | synapses µJ |
|---|---:|---:|---|---:|---:|
| muxlin (validation-selected: ADD, lr 0.02, 2 layers) | 91.99 ± 0.10 | 98.56 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 9.44 | 7.63 |
| muxlin (default: OR, lr 0.01, 2 layers) | 91.90 ± 0.25 | 98.55 | 99.24 / 85.74 / 96.38 / 79.24 / 98.89 | 8.89 | 7.10 |
| NSMT rerun | 91.58 ± 0.05 | 98.41 | 99.16 / 84.31 / 95.65 / 79.89 / 98.89 | 23.86 | 19.06 |
| Spikformer-style (tuned, lr 0.005) | 91.78 ± 0.25 | 98.48 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 9.61 | 7.95 |
| Spikformer-style (default, lr 0.01) | 92.22 ± 0.17 | 98.55 | 99.24 / 85.28 / 96.04 / 81.38 / 99.17 | 9.55 | 7.87 |
| NSMT paper (Table 5) | 91.91 | 98.45 | — | — | 19.05 |

| A vs B | A − B %p (seed 42, 7, 2026) | mean | verdict | energy A/B |
|---|---|---:|---|---:|
| muxlin (selected) vs NSMT rerun | +0.24, +0.56, +0.41 | +0.40 | no clear difference | 0.40 |
| muxlin (selected) vs Spikformer-style (tuned) | -0.22, +0.39, +0.44 | +0.20 | no clear difference | 0.98 |
| muxlin (selected) vs Spikformer-style (default) | -0.53, -0.12, -0.06 | -0.24 | no clear difference | 0.99 |
| NSMT rerun vs Spikformer-style (default) | -0.78, -0.68, -0.47 | -0.64 | worse | 2.50 |

## AD test (TSLib procedure; seeds 42, 7, 2026, mean; NSMT was not rerun here)

| data | model | AUC-PR % | AUC-ROC % | raw F1 % | point-adjusted F1 % | energy µJ |
|---|---|---:|---:|---:|---:|---:|
| MSL | muxlin (validation-selected: OR, lr 0.002, 2 layers) | 14.67 | 61.23 | 7.81 | 41.12 | 14.64 |
| MSL | muxlin (default: OR, lr 5e-4, 2 layers) | 14.89 | 61.57 | 7.89 | 43.04 | 14.48 |
| MSL | Spikformer-style (tuned, lr 0.00025) | 14.46 | 60.43 | 7.66 | 67.90 | 14.50 |
| MSL | NSMT paper (Table 9) | — | — | — | 76.51 | — |
| MSL | no training: random | 10.53 | 49.97 | 1.82 | 84.60 | 0 |
| MSL | no training: win-mean | 14.96 | 64.07 | 4.47 | 81.79 | 0 |
| MSL | no training: train-dev | 14.02 | 53.27 | 6.43 | 82.06 | 0 |
| SMAP | muxlin (validation-selected: ADD, lr 0.002, 1 layers) | 11.81 | 44.92 | 3.03 | 68.32 | 7.06 |
| SMAP | muxlin (default: OR, lr 5e-4, 2 layers) | 11.88 | 45.42 | 2.76 | 68.15 | 7.22 |
| SMAP | Spikformer-style (tuned, lr 0.002) | 11.59 | 44.89 | 2.47 | 67.64 | 7.18 |
| SMAP | NSMT paper (Table 9) | — | — | — | 75.75 | — |
| SMAP | no training: random | 12.79 | 49.99 | 1.86 | 93.75 | 0 |
| SMAP | no training: win-mean | 10.84 | 43.50 | 1.18 | 67.39 | 0 |
| SMAP | no training: train-dev | 10.56 | 41.31 | 1.23 | 67.07 | 0 |
| PSM | muxlin (validation-selected: ADD, lr 0.001, 3 layers) | 38.69 | 59.45 | 5.49 | 93.78 | 8.66 |
| PSM | muxlin (default: OR, lr 5e-4, 2 layers) | 39.34 | 60.12 | 5.30 | 92.60 | 7.56 |
| PSM | Spikformer-style (tuned, lr 0.0005) | 38.76 | 59.54 | 5.15 | 93.75 | 7.79 |
| PSM | NSMT paper (Table 9) | — | — | — | 95.46 | — |
| PSM | no training: random | 27.75 | 50.01 | 1.93 | 95.89 | 0 |
| PSM | no training: win-mean | 38.12 | 58.53 | 4.19 | 92.42 | 0 |
| PSM | no training: train-dev | 47.26 | 65.81 | 3.30 | 88.32 | 0 |

| muxlin vs (AUC-PR; rule R >= +2 % and >= 6 / 9 higher) | R % | higher pairs | verdict |
|---|---:|---:|---|
| selected vs Spikformer-style (tuned) | +1.05 | 6/9 | no clear difference |
| default vs Spikformer-style (tuned) | +2.33 | 9/9 | better |

## CinC fold 0 test (support only, not a main result; 30 epochs; the paper has no CinC result)

| model | CinC score % | macro-F1 % | energy µJ |
|---|---:|---:|---:|
| muxlin (OR, lr 1e-3, 2 layers) | 68.52 ± 0.81 | 64.35 | 129.4 |
| NSMT rerun (LIF) | 61.76 ± 1.15 | 59.01 | 3419.3 |
| Spikformer-style (tuned, lr 2e-3) | 66.36 ± 1.04 | 62.38 | 131.5 |
| Spikformer-style (default, lr 1e-3) | 65.60 ± 1.21 | 61.94 | 127.4 |

| A vs B | A − B %p (seed 42, 7, 2026) | mean | verdict | energy A/B |
|---|---|---:|---|---:|
| muxlin vs NSMT rerun | +7.55, +7.24, +5.49 | +6.76 | better | 0.04 |
| muxlin vs Spikformer-style (tuned) | +2.86, +2.54, +1.09 | +2.16 | better | 0.98 |
| NSMT rerun vs Spikformer-style (default) | -5.30, -4.14, -2.06 | -3.83 | worse | 26.85 |

