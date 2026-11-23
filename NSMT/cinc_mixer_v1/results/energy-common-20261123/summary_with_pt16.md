# ETT energy under one rule: syops as published (µJ per input window, all 7 channels)

Rules and sources: module docstring of common_table.py. Ours / SF-de: seed means; NSMT: seed 42; SpikF: one trained checkpoint.

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | 4-cell mean |
|---|--:|--:|--:|--:|--:|
| LIF SpkMixer, first-chain final (patch 16) | 5.92 | 21.31 | 6.58 | 21.94 | 13.94 |
| LIF SpkMixer, extension final (patch 24) | 4.50 | 14.76 | 32.44 | 73.32 | 31.26 |
| SF-de patch 8 (criteria SF-de) | 10.11 | 40.92 | 10.77 | 41.43 | 25.81 |
| SF-de patch 16 | 5.91 | 21.32 | 6.17 | 21.55 | 13.74 |
| NSMT (10-14 reproduction, seed 42) | 33.41 | 44.90 | 38.46 | 46.72 | 40.87 |
| SpikF (H96: output layer resized, see count_spikf.py) | 819.00 | 1050.47 | 818.66 | 1050.10 | 934.56 |

With the f-LIF-paper neuron term added (our runs only; the rule used for the verdicts so far):

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | neuron term share |
|---|--:|--:|--:|--:|---|
| LIF SpkMixer, first-chain final (patch 16) | 7.54 | 22.93 | 8.58 | 23.94 | 21.5 %, 7.1 %, 23.4 %, 8.3 % |
| LIF SpkMixer, extension final (patch 24) | 5.58 | 15.84 | 41.06 | 81.93 | 19.4 %, 6.8 %, 21.0 %, 10.5 % |
| SF-de patch 8 (criteria SF-de) | 13.84 | 44.64 | 14.80 | 45.44 | 27.0 %, 8.3 %, 27.3 %, 8.8 % |
| SF-de patch 16 | 7.79 | 23.19 | 8.19 | 23.55 | 24.1 %, 8.1 %, 24.6 %, 8.5 % |

Ratio first-chain final / other model, per cell (syops as published):

| vs | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean of cell ratios |
|---|--:|--:|--:|--:|--:|
| LIF SpkMixer, extension final (patch 24) | 1.315 | 1.444 | 0.203 | 0.299 | 0.815 |
| SF-de patch 8 (criteria SF-de) | 0.585 | 0.521 | 0.611 | 0.530 | 0.562 |
| SF-de patch 16 | 1.001 | 1.000 | 1.065 | 1.018 | 1.021 |
| NSMT (10-14 reproduction, seed 42) | 0.177 | 0.475 | 0.171 | 0.470 | 0.323 |
| SpikF (H96: output layer resized, see count_spikf.py) | 0.007 | 0.020 | 0.008 | 0.021 | 0.014 |
