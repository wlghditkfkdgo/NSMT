# Energy by part (CinC fold 0, one test window; syops per-layer files, 0.9 pJ / AC, 4.6 pJ / MAC; neuron state updates from RESULT neuron_model_fft_uJ) — mean of seeds 42 / 7 / 2026, µJ

| part | muxlin | Spikformer-style (lr 2e-3) |
|---|---:|---:|
| embedding Linear (real input, MAC, x T_h) | 52.9 (41 %) | 52.9 (40 %) |
| BatchNorm (MAC) | 14.9 (12 %) | 11.6 (9 %) |
| token mixer / attention projections (AC) | 16.5 (13 %) | 36.7 (28 %) |
| fusion Linear (AC) | 14.2 (11 %) | 0.0 (0 %) |
| MLP (AC) | 10.4 (8 %) | 11.5 (9 %) |
| head | 0.0 (0 %) | 0.0 (0 %) |
| neuron input ACs (syops) | 1.5 (1 %) | 1.3 (1 %) |
| neuron state updates | 19.1 (15 %) | 17.5 (13 %) |
| total (sum) / logged total_fft_uJ | 129.4 / 129.4 | 131.5 / 131.5 |

What-if (same checkpoints, accounting only, not a new model):

| accounting | muxlin | Spikformer-style (lr 2e-3) | ratio |
|---|---:|---:|---:|
| as logged (embedding x T_h, BN counted) | 129.4 | 131.5 | 0.98 |
| embedding computed once (its T_h repeats are identical) | 89.7 | 91.8 | 0.98 |
| + BN folded into the preceding Linear | 74.8 | 80.3 | 0.93 |
