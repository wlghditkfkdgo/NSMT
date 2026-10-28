# PPG-DaLiA heart-rate regression `smoke-ppg-20261028` (Log-NCDE protocol, seed-0 split, causal models) — summary

Test MSE x 1e-2 on the scaled labels and MAE in BPM; best-validation weights; mean (seed 42 / 7 / 2026). Literature numbers use other (unseeded) splits and mostly non-causal settings: reference only.

| model | MSE x 1e-2 | MAE BPM | + smoothing MSE x 1e-2 | total µJ (FFT) | params M |
|---|---|---|---|---|---|
| P0 token only (window 1) | incomplete | incomplete | incomplete | incomplete | incomplete |
| P1 Spikformer-style, window 3 (8 s) | incomplete | incomplete | incomplete | incomplete | incomplete |
| P2 window 15 (32 s) | incomplete | incomplete | incomplete | incomplete | incomplete |
| P3 all past | incomplete | incomplete | incomplete | incomplete | incomplete |
| P4 Neo-mixer forward, LIF | incomplete | incomplete | incomplete | incomplete | incomplete |
| P4 Neo-mixer forward, f-LIF | incomplete | incomplete | incomplete | incomplete | incomplete |
| P4 Neo-mixer forward, population LIF | incomplete | incomplete | incomplete | incomplete | incomplete |
| P4 Neo-mixer forward, population f-LIF | incomplete | incomplete | incomplete | incomplete | incomplete |
| P5 window 3 + T = N neocortex, LIF | incomplete | incomplete | incomplete | incomplete | incomplete |
| P5 window 3 + T = N neocortex, f-LIF | incomplete | incomplete | incomplete | incomplete | incomplete |
| P5 window 3 + T = N neocortex, population LIF | incomplete | incomplete | incomplete | incomplete | incomplete |
| P5 window 3 + T = N neocortex, population f-LIF | incomplete | incomplete | incomplete | incomplete | incomplete |
| P6 window 3 + memoryless LIF neocortex | incomplete | incomplete | incomplete | incomplete | incomplete |
| GRU (non-spiking) | incomplete | incomplete | incomplete | — | incomplete |

## Pre-registered comparisons (relative MSE difference per seed; better = mean <= -10 % and 3 / 3 seeds)

| question | A | B | (A − B) / B per seed | verdict |
|---|---|---|---|---|
| V1 state beyond the window (main) | P5 window 3 + T = N neocortex, LIF | P6 window 3 + memoryless LIF neocortex | — | incomplete |
| V1 state beyond the window (main) | P5 window 3 + T = N neocortex, f-LIF | P6 window 3 + memoryless LIF neocortex | — | incomplete |
| V1 state beyond the window (main) | P5 window 3 + T = N neocortex, population LIF | P6 window 3 + memoryless LIF neocortex | — | incomplete |
| V1 state beyond the window (main) | P5 window 3 + T = N neocortex, population f-LIF | P6 window 3 + memoryless LIF neocortex | — | incomplete |
| V2 beyond smoothing | P5 window 3 + T = N neocortex, LIF | P1 + smoothing | — | incomplete |
| V2 beyond smoothing | P5 window 3 + T = N neocortex, f-LIF | P1 + smoothing | — | incomplete |
| V2 beyond smoothing | P5 window 3 + T = N neocortex, population LIF | P1 + smoothing | — | incomplete |
| V2 beyond smoothing | P5 window 3 + T = N neocortex, population f-LIF | P1 + smoothing | — | incomplete |
| V3 state only vs attention 8 s | P4 Neo-mixer forward, LIF | P1 Spikformer-style, window 3 (8 s) | — | incomplete |
| V3 state only vs attention all past | P4 Neo-mixer forward, LIF | P3 all past | — | incomplete |
| V3 state only vs attention 8 s | P4 Neo-mixer forward, f-LIF | P1 Spikformer-style, window 3 (8 s) | — | incomplete |
| V3 state only vs attention all past | P4 Neo-mixer forward, f-LIF | P3 all past | — | incomplete |
| V3 state only vs attention 8 s | P4 Neo-mixer forward, population LIF | P1 Spikformer-style, window 3 (8 s) | — | incomplete |
| V3 state only vs attention all past | P4 Neo-mixer forward, population LIF | P3 all past | — | incomplete |
| V3 state only vs attention 8 s | P4 Neo-mixer forward, population f-LIF | P1 Spikformer-style, window 3 (8 s) | — | incomplete |
| V3 state only vs attention all past | P4 Neo-mixer forward, population f-LIF | P3 all past | — | incomplete |
| V4 context for attention | P2 window 15 (32 s) | P1 Spikformer-style, window 3 (8 s) | — | incomplete |
| V4 context for attention | P3 all past | P1 Spikformer-style, window 3 (8 s) | — | incomplete |
| V4 context for attention | P1 Spikformer-style, window 3 (8 s) | P0 token only (window 1) | — | incomplete |
| V5 recurrent reference | GRU (non-spiking) | P1 Spikformer-style, window 3 (8 s) | — | incomplete |
| neuron (report) | P5 window 3 + T = N neocortex, f-LIF | P5 window 3 + T = N neocortex, LIF | — | incomplete |
| neuron (report) | P4 Neo-mixer forward, f-LIF | P4 Neo-mixer forward, LIF | — | incomplete |
| neuron (report) | P5 window 3 + T = N neocortex, population LIF | P5 window 3 + T = N neocortex, LIF | — | incomplete |
| neuron (report) | P4 Neo-mixer forward, population LIF | P4 Neo-mixer forward, LIF | — | incomplete |
| neuron (report) | P5 window 3 + T = N neocortex, population f-LIF | P5 window 3 + T = N neocortex, LIF | — | incomplete |
| neuron (report) | P4 Neo-mixer forward, population f-LIF | P4 Neo-mixer forward, LIF | — | incomplete |

