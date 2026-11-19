# SNN models at look-back 96 on ETT (NSMT Table 14 / SpikF Table 12) vs the current SpkMixer

Rules and sources: module docstring of compare_snn_lb96.py. Ours: seed mean ± std (42 / 7 / 2026).
Transcription verified against both papers' text (--verify).

## Test MSE / MAE

| model | ETTh2 H96 MSE / MAE | ETTh2 H720 MSE / MAE | ETTm2 H96 MSE / MAE | ETTm2 H720 MSE / MAE | 4-cell mean MSE / MAE |
|---|---|---|---|---|---|
| LIF SpkMixer (ours) | 0.2952 ± 0.0029 / 0.3381 | 0.4218 ± 0.0031 / 0.4384 | 0.1775 ± 0.0003 / 0.2547 | 0.4034 ± 0.0002 / 0.3955 | 0.3245 / 0.3567 |
| SF-de (ours) | 0.2921 ± 0.0031 / 0.3360 | 0.4229 ± 0.0050 / 0.4377 | 0.1768 ± 0.0004 / 0.2538 | 0.4036 ± 0.0010 / 0.3953 | 0.3238 / 0.3557 |
| NSMT | 0.288 / 0.334 | 0.409 / 0.429 | 0.175 / 0.251 | 0.399 / 0.393 | 0.3177 / 0.3518 |
| SpikF | 0.290 / 0.336 | 0.416 / 0.436 | 0.175 / 0.251 | 0.405 / 0.397 | 0.3215 / 0.3550 |
| SpikeRNN | 0.295 / 0.335 | 0.428 / 0.438 | 0.176 / 0.251 | 0.409 / 0.396 | 0.3270 / 0.3550 |
| iSpikformer | 0.308 / 0.349 | 0.438 / 0.439 | 0.178 / 0.256 | 0.406 / 0.398 | 0.3325 / 0.3605 |

## Ours vs each paper model (R = mean of the 4 per-cell relative differences; negative = ours better)

| ours | vs | R MSE % | cells lower (MSE) | R MAE % | cells lower (MAE) | 4-cell mean MSE difference % | per-cell MSE % (ETTh2 H96, ETTh2 H720, ETTm2 H96, ETTm2 H720) |
|---|---|--:|--:|--:|--:|--:|---|
| LIF SpkMixer (ours) | NSMT | +2.05 | 0/4 | +1.39 | 0/4 | +2.12 | +2.50, +3.13, +1.45, +1.11 |
| LIF SpkMixer (ours) | SpikF | +1.06 | 1/4 | +0.57 | 1/4 | +0.93 | +1.79, +1.40, +1.45, -0.38 |
| LIF SpkMixer (ours) | SpikeRNN | -0.47 | 2/4 | +0.60 | 1/4 | -0.77 | +0.07, -1.45, +0.87, -1.36 |
| LIF SpkMixer (ours) | iSpikformer | -2.19 | 4/4 | -1.09 | 4/4 | -2.41 | -4.16, -3.70, -0.26, -0.63 |
| SF-de (ours) | NSMT | +1.75 | 0/4 | +1.08 | 0/4 | +1.92 | +1.42, +3.39, +1.02, +1.16 |
| SF-de (ours) | SpikF | +0.76 | 1/4 | +0.27 | 2/4 | +0.73 | +0.72, +1.65, +1.02, -0.34 |
| SF-de (ours) | SpikeRNN | -0.76 | 3/4 | +0.29 | 2/4 | -0.96 | -0.98, -1.20, +0.45, -1.31 |
| SF-de (ours) | iSpikformer | -2.47 | 4/4 | -1.39 | 4/4 | -2.60 | -5.16, -3.46, -0.68, -0.58 |

## Seeds of the LIF SpkMixer that beat each paper value (MSE, per cell; report)

| cell | seeds 42 / 7 / 2026 | NSMT | SpikF | SpikeRNN | iSpikformer |
|---|---|--:|--:|--:|--:|
| ETTh2 H96 | 0.2985, 0.2914, 0.2956 | 0/3 | 0/3 | 1/3 | 3/3 |
| ETTh2 H720 | 0.4255, 0.4178, 0.4221 | 0/3 | 0/3 | 3/3 | 3/3 |
| ETTm2 H96 | 0.1773, 0.1774, 0.1779 | 0/3 | 0/3 | 0/3 | 3/3 |
| ETTm2 H720 | 0.4034, 0.4033, 0.4037 | 0/3 | 3/3 | 3/3 | 3/3 |

## Cells of the papers not run here (ETTh1, ETTm1, H192, H336; user decision 12): NSMT Table 14 averages over 4 horizons (MSE)

| dataset | SpikF | SpikeRNN | iSpikformer | NSMT |
|---|--:|--:|--:|--:|
| ETTh1 | 0.440 | 0.452 | 0.473 | 0.435 |
| ETTh2 | 0.372 | 0.380 | 0.392 | 0.365 |
| ETTm1 | 0.388 | 0.389 | 0.402 | 0.380 |
| ETTm2 | 0.281 | 0.283 | 0.283 | 0.279 |
