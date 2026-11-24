# Full ETT table, input 96: both final LIF SpkMixer configurations vs the SpikF paper's SNN rows and the NSMT paper's DNN rows

Rules and sources: module docstring of summarize_full16.py. Ours: seed mean of 42 / 7 / 2026 (test values of the best-validation weights). ETTh1 / ETTm1 use the settings chosen on ETTh2 / ETTm2 (no selection on them). Paper rows: single published numbers.

## ETTh1 (test MSE / MAE)

| H | first-chain (pt 16) | extension (pt 24) | SpikF | SpikeRNN | iSpikformer | PatchTST | NSMT |
|---|---|---|---|---|---|---|---|
| 96 | 0.3775 / 0.3897 | 0.3780 / 0.3915 | 0.379 / 0.391 | 0.392 / 0.395 | 0.410 / 0.408 | 0.414 / 0.419 | 0.378 / 0.386 |
| 192 | 0.4382 / 0.4239 | 0.4404 / 0.4250 | 0.432 / 0.421 | 0.437 / 0.424 | 0.459 / 0.438 | 0.460 / 0.445 | 0.426 / 0.417 |
| 336 | 0.4818 / 0.4451 | 0.4891 / 0.4483 | 0.473 / 0.441 | 0.482 / 0.447 | 0.514 / 0.461 | 0.501 / 0.466 | 0.473 / 0.441 |
| 720 | 0.4925 / 0.4674 | 0.4952 / 0.4673 | 0.474 / 0.459 | 0.499 / 0.471 | 0.511 / 0.476 | 0.500 / 0.488 | 0.461 / 0.454 |
| Avg | 0.4475 / 0.4315 | 0.4507 / 0.4330 | 0.440 / 0.428 | 0.452 / 0.434 | 0.473 / 0.446 | 0.469 / 0.455 | 0.435 / 0.425 |

## ETTh2 (test MSE / MAE)

| H | first-chain (pt 16) | extension (pt 24) | SpikF | SpikeRNN | iSpikformer | PatchTST | NSMT |
|---|---|---|---|---|---|---|---|
| 96 | 0.2902 / 0.3347 | 0.2887 / 0.3327 | 0.290 / 0.336 | 0.295 / 0.335 | 0.308 / 0.349 | 0.302 / 0.348 | 0.288 / 0.334 |
| 192 | 0.3633 / 0.3813 | 0.3624 / 0.3806 | 0.367 / 0.385 | 0.375 / 0.387 | 0.382 / 0.397 | 0.388 / 0.400 | 0.358 / 0.380 |
| 336 | 0.4111 / 0.4187 | 0.4097 / 0.4175 | 0.414 / 0.420 | 0.422 / 0.422 | 0.439 / 0.433 | 0.426 / 0.433 | 0.403 / 0.412 |
| 720 | 0.4151 / 0.4327 | 0.4141 / 0.4318 | 0.416 / 0.436 | 0.428 / 0.438 | 0.438 / 0.439 | 0.431 / 0.446 | 0.409 / 0.429 |
| Avg | 0.3699 / 0.3919 | 0.3687 / 0.3906 | 0.372 / 0.394 | 0.380 / 0.396 | 0.392 / 0.407 | 0.387 / 0.407 | 0.365 / 0.389 |

## ETTm1 (test MSE / MAE)

| H | first-chain (pt 16) | extension (pt 24) | SpikF | SpikeRNN | iSpikformer | PatchTST | NSMT |
|---|---|---|---|---|---|---|---|
| 96 | 0.3171 / 0.3457 | 0.3204 / 0.3476 | 0.317 / 0.345 | 0.326 / 0.351 | 0.337 / 0.361 | 0.329 / 0.367 | 0.317 / 0.345 |
| 192 | 0.3694 / 0.3715 | 0.3715 / 0.3738 | 0.372 / 0.372 | 0.376 / 0.377 | 0.380 / 0.383 | 0.367 / 0.385 | 0.364 / 0.371 |
| 336 | 0.3980 / 0.3930 | 0.4071 / 0.3971 | 0.401 / 0.394 | 0.396 / 0.394 | 0.415 / 0.406 | 0.399 / 0.410 | 0.390 / 0.391 |
| 720 | 0.4645 / 0.4317 | 0.4737 / 0.4363 | 0.461 / 0.430 | 0.459 / 0.433 | 0.476 / 0.443 | 0.454 / 0.439 | 0.446 / 0.425 |
| Avg | 0.3872 / 0.3855 | 0.3932 / 0.3887 | 0.388 / 0.385 | 0.389 / 0.389 | 0.402 / 0.398 | 0.387 / 0.400 | 0.380 / 0.383 |

## ETTm2 (test MSE / MAE)

| H | first-chain (pt 16) | extension (pt 24) | SpikF | SpikeRNN | iSpikformer | PatchTST | NSMT |
|---|---|---|---|---|---|---|---|
| 96 | 0.1748 / 0.2525 | 0.1745 / 0.2521 | 0.175 / 0.251 | 0.176 / 0.251 | 0.178 / 0.256 | 0.175 / 0.259 | 0.175 / 0.251 |
| 192 | 0.2405 / 0.2965 | 0.2408 / 0.2965 | 0.242 / 0.296 | 0.243 / 0.296 | 0.243 / 0.298 | 0.241 / 0.302 | 0.241 / 0.297 |
| 336 | 0.3019 / 0.3360 | 0.3007 / 0.3351 | 0.302 / 0.336 | 0.302 / 0.335 | 0.304 / 0.338 | 0.305 / 0.343 | 0.302 / 0.335 |
| 720 | 0.4018 / 0.3940 | 0.3999 / 0.3937 | 0.405 / 0.397 | 0.409 / 0.396 | 0.406 / 0.398 | 0.402 / 0.400 | 0.399 / 0.393 |
| Avg | 0.2797 / 0.3197 | 0.2790 / 0.3194 | 0.281 / 0.320 | 0.283 / 0.319 | 0.283 / 0.322 | 0.281 / 0.326 | 0.279 / 0.319 |

## Judgment (16-cell means; margin = ref − ours, ≥ +0.0010 needed for MSE and MAE)

### first-chain final (patch 16): 16-cell MSE / MAE **0.3711 / 0.3821** (the 4 cells of 11-23: 0.3205 / 0.3535)

| reference | role | ref 16-cell MSE / MAE | margin MSE / MAE | passes | cells with lower MSE / MAE (of 16) |
|---|---|---|---|---|---|
| SpikF | SNN (SpikF paper) | 0.3700 / 0.3819 | -0.0011 / -0.0003 | no | 10 / 9 |
| SpikeRNN | SNN (SpikF paper) | 0.3761 / 0.3845 | +0.0050 / +0.0024 | yes | 13 / 13 |
| iSpikformer | SNN (SpikF paper) | 0.3875 / 0.3928 | +0.0164 / +0.0106 | yes | 16 / 16 |
| Informer | DNN (NSMT Table 13) | 1.9607 / 1.0169 | +1.5896 / +0.6347 | yes | 16 / 16 |
| FEDformer | DNN (NSMT Table 13) | 0.4074 / 0.4276 | +0.0363 / +0.0454 | yes | 13 / 16 |
| PatchTST | DNN (NSMT Table 13) | 0.3809 / 0.3969 | +0.0098 / +0.0147 | yes | 14 / 16 |
| ETSformer | DNN (NSMT Table 13) | 0.4256 / 0.4320 | +0.0545 / +0.0499 | yes | 16 / 16 |
| Autoformer | DNN (NSMT Table 13) | 0.4651 / 0.4586 | +0.0940 / +0.0765 | yes | 16 / 16 |
| NSMT | report only (own paper) | 0.3644 / 0.3788 | -0.0067 / -0.0033 | no | 4 / 1 |

Energy (syops): below SpikF in **16 / 16** cells (needed 16); ours / SpikF mean of cell ratios 0.0130, largest 0.0210 (ETTm1 H720).
Accuracy rows (3 SNN + 5 DNN): NOT all pass: SpikF. **Verdict: NOT PASSED**

### extension final (patch 24): 16-cell MSE / MAE **0.3729 / 0.3829** (the 4 cells of 11-23: 0.3193 / 0.3526)

| reference | role | ref 16-cell MSE / MAE | margin MSE / MAE | passes | cells with lower MSE / MAE (of 16) |
|---|---|---|---|---|---|
| SpikF | SNN (SpikF paper) | 0.3700 / 0.3819 | -0.0029 / -0.0011 | no | 10 / 6 |
| SpikeRNN | SNN (SpikF paper) | 0.3761 / 0.3845 | +0.0032 / +0.0016 | yes | 12 / 9 |
| iSpikformer | SNN (SpikF paper) | 0.3875 / 0.3928 | +0.0146 / +0.0098 | yes | 16 / 16 |
| Informer | DNN (NSMT Table 13) | 1.9607 / 1.0169 | +1.5878 / +0.6339 | yes | 16 / 16 |
| FEDformer | DNN (NSMT Table 13) | 0.4074 / 0.4276 | +0.0345 / +0.0446 | yes | 13 / 16 |
| PatchTST | DNN (NSMT Table 13) | 0.3809 / 0.3969 | +0.0080 / +0.0139 | yes | 13 / 16 |
| ETSformer | DNN (NSMT Table 13) | 0.4256 / 0.4320 | +0.0527 / +0.0491 | yes | 16 / 16 |
| Autoformer | DNN (NSMT Table 13) | 0.4651 / 0.4586 | +0.0922 / +0.0757 | yes | 16 / 16 |
| NSMT | report only (own paper) | 0.3644 / 0.3788 | -0.0085 / -0.0041 | no | 3 / 2 |

Energy (syops): below SpikF in **16 / 16** cells (needed 16); ours / SpikF mean of cell ratios 0.0308, largest 0.0711 (ETTm1 H720).
Accuracy rows (3 SNN + 5 DNN): NOT all pass: SpikF. **Verdict: NOT PASSED**

## Energy per cell (syops µJ per input window; ours seed mean, SpikF one checkpoint, NSMT seed 42)

| cell | first-chain | extension | SpikF | NSMT | first / SpikF | ext / SpikF | first / NSMT | ext / NSMT |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| ETTh1 H96 | 5.97 | 4.56 | 819.12 | 18.76 | 0.0073 | 0.0056 | 0.318 | 0.243 |
| ETTh1 H192 | 8.35 | 6.13 | 854.73 | 19.47 | 0.0098 | 0.0072 | 0.429 | 0.315 |
| ETTh1 H336 | 11.89 | 8.52 | 908.15 | 21.58 | 0.0131 | 0.0094 | 0.551 | 0.395 |
| ETTh1 H720 | 21.39 | 14.83 | 1050.59 | 26.16 | 0.0204 | 0.0141 | 0.818 | 0.567 |
| ETTh2 H96 | 5.92 | 4.50 | 819.00 | 33.41 | 0.0072 | 0.0055 | 0.177 | 0.135 |
| ETTh2 H192 | 8.27 | 6.07 | 854.61 | 34.39 | 0.0097 | 0.0071 | 0.241 | 0.177 |
| ETTh2 H336 | 11.84 | 8.45 | 908.03 | 36.73 | 0.0130 | 0.0093 | 0.322 | 0.230 |
| ETTh2 H720 | 21.31 | 14.76 | 1050.47 | 44.90 | 0.0203 | 0.0141 | 0.475 | 0.329 |
| ETTm1 H96 | 6.65 | 33.10 | 819.06 | 40.55 | 0.0081 | 0.0404 | 0.164 | 0.816 |
| ETTm1 H192 | 9.05 | 39.79 | 854.67 | 42.19 | 0.0106 | 0.0466 | 0.215 | 0.943 |
| ETTm1 H336 | 12.60 | 49.39 | 908.09 | 44.57 | 0.0139 | 0.0544 | 0.283 | 1.108 |
| ETTm1 H720 | 22.09 | 74.67 | 1050.53 | 49.80 | 0.0210 | 0.0711 | 0.444 | 1.499 |
| ETTm2 H96 | 6.58 | 32.44 | 818.66 | 38.46 | 0.0080 | 0.0396 | 0.171 | 0.844 |
| ETTm2 H192 | 8.94 | 38.78 | 854.26 | 38.80 | 0.0105 | 0.0454 | 0.230 | 1.000 |
| ETTm2 H336 | 12.49 | 48.25 | 907.68 | 41.45 | 0.0138 | 0.0532 | 0.301 | 1.164 |
| ETTm2 H720 | 21.94 | 73.32 | 1050.10 | 46.72 | 0.0209 | 0.0698 | 0.470 | 1.569 |
| 16-cell mean | 12.20 | 28.60 | 907.98 | 36.12 | | | | |

## Seed spread (std of test MSE over the 3 seeds, mean over the 16 cells)

- first-chain final (patch 16): 0.0020
- extension final (patch 24): 0.0021
