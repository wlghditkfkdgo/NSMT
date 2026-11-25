# ETTh1 / ETTm1 retuned on their own validation: full ETT table, input 96

Rules: summarize_rt.py / rt_plan.py docstrings. Ours: seed mean of 42 / 7 / 2026 (test values of the best-validation weights). ETTh2 / ETTm2 cells are the 11-24 values (unchanged). Paper rows: single published numbers.

## Selection paths on ETTh1 / ETTm1 (validation score of the winner; 1 = kept the current value)

### shared stages

| stage | ETTh1 | ETTm1 |
|---|---|---|
| lr (LR) | 0.0005 → **0.00025** (0.9981) | 0.001 → **0.0005** (0.9961) |
| layers (LAYERS) | 2 → **1** (0.9999) | 2 → **2** (1.0000) |

### first chain (patch <= 16, wd <= 0.1)

| stage | ETTh1 | ETTm1 |
|---|---|---|
| patch (PATCH) | 8 → **8** (1.0000) | 8 → **16** (0.9951) |
| dmodel (EMB) | 32 → **32** (1.0000) | 64 → **32** (0.9995) |
| mlp (MLPR) | 1 → **4** (0.9994) | 1 → **0.5** (0.9985) |
| blocks (BLOCKS) | 1 → **1** (1.0000) | 1 → **3** (0.9991) |
| wd (WD) | 0.06 → **0.1** (0.9997) | 0.06 → **0.06** (1.0000) |

### extension (patch <= 32, wd <= 0.06)

| stage | ETTh1 | ETTm1 |
|---|---|---|
| patch (PATCH) | 8 → **32** (0.9999) | 8 → **32** (0.9946) |
| dmodel (EMB) | 32 → **256** (0.9976) | 64 → **64** (1.0000) |
| mlp (MLPR) | 1 → **1** (1.0000) | 1 → **1** (1.0000) |
| blocks (BLOCKS) | 1 → **1** (1.0000) | 1 → **2** (0.9996) |
| wd (WD) | 0.06 → **0.06** (1.0000) | 0.06 → **0.06** (1.0000) |

Final configurations: first chain ETTh1: LR 0.00025, LAYERS 1, PATCH 8, EMB 32, MLPR 4, BLOCKS 1, WD 0.1 (tag spkmx_lif_muxlin_iand_lr0.00025_L1_e32_mlp4_wd0.1); first chain ETTm1: LR 0.0005, LAYERS 2, PATCH 16, EMB 32, MLPR 0.5, BLOCKS 3, WD 0.06 (tag spkmx_lif_muxlin_iand_lr0.0005_L2_pt16_e32_mlp0.5_blk3); extension ETTh1: LR 0.00025, LAYERS 1, PATCH 32, EMB 256, MLPR 1, BLOCKS 1, WD 0.06 (tag spkmx_lif_muxlin_iand_lr0.00025_L1_pt32_e256_mlp1); extension ETTm1: LR 0.0005, LAYERS 2, PATCH 32, EMB 64, MLPR 1, BLOCKS 2, WD 0.06 (tag spkmx_lif_muxlin_iand_lr0.0005_L2_pt32_blk2)

## ETTh1 (test MSE / MAE)

| H | first chain | extension | SpikF | SpikeRNN | iSpikformer | PatchTST |
|---|---|---|---|---|---|---|
| 96 | 0.3833 / 0.3907 | 0.3764 / 0.3920 | 0.379 / 0.391 | 0.392 / 0.395 | 0.410 / 0.408 | 0.414 / 0.419 |
| 192 | 0.4396 / 0.4213 | 0.4416 / 0.4272 | 0.432 / 0.421 | 0.437 / 0.424 | 0.459 / 0.438 | 0.460 / 0.445 |
| 336 | 0.4820 / 0.4416 | 0.4842 / 0.4465 | 0.473 / 0.441 | 0.482 / 0.447 | 0.514 / 0.461 | 0.501 / 0.466 |
| 720 | 0.4788 / 0.4590 | 0.4924 / 0.4676 | 0.474 / 0.459 | 0.499 / 0.471 | 0.511 / 0.476 | 0.500 / 0.488 |
| Avg | 0.4459 / 0.4281 | 0.4487 / 0.4333 | 0.440 / 0.428 | 0.452 / 0.434 | 0.473 / 0.446 | 0.469 / 0.455 |

## ETTh2 (test MSE / MAE) — unchanged from 11-24

| H | first chain | extension | SpikF | SpikeRNN | iSpikformer | PatchTST |
|---|---|---|---|---|---|---|
| 96 | 0.2902 / 0.3347 | 0.2887 / 0.3327 | 0.290 / 0.336 | 0.295 / 0.335 | 0.308 / 0.349 | 0.302 / 0.348 |
| 192 | 0.3633 / 0.3813 | 0.3624 / 0.3806 | 0.367 / 0.385 | 0.375 / 0.387 | 0.382 / 0.397 | 0.388 / 0.400 |
| 336 | 0.4111 / 0.4187 | 0.4097 / 0.4175 | 0.414 / 0.420 | 0.422 / 0.422 | 0.439 / 0.433 | 0.426 / 0.433 |
| 720 | 0.4151 / 0.4327 | 0.4141 / 0.4318 | 0.416 / 0.436 | 0.428 / 0.438 | 0.438 / 0.439 | 0.431 / 0.446 |
| Avg | 0.3699 / 0.3919 | 0.3687 / 0.3906 | 0.372 / 0.394 | 0.380 / 0.396 | 0.392 / 0.407 | 0.387 / 0.407 |

## ETTm1 (test MSE / MAE)

| H | first chain | extension | SpikF | SpikeRNN | iSpikformer | PatchTST |
|---|---|---|---|---|---|---|
| 96 | 0.3198 / 0.3458 | 0.3189 / 0.3451 | 0.317 / 0.345 | 0.326 / 0.351 | 0.337 / 0.361 | 0.329 / 0.367 |
| 192 | 0.3730 / 0.3734 | 0.3724 / 0.3725 | 0.372 / 0.372 | 0.376 / 0.377 | 0.380 / 0.383 | 0.367 / 0.385 |
| 336 | 0.4031 / 0.3941 | 0.4077 / 0.3957 | 0.401 / 0.394 | 0.396 / 0.394 | 0.415 / 0.406 | 0.399 / 0.410 |
| 720 | 0.4676 / 0.4321 | 0.4732 / 0.4344 | 0.461 / 0.430 | 0.459 / 0.433 | 0.476 / 0.443 | 0.454 / 0.439 |
| Avg | 0.3909 / 0.3863 | 0.3930 / 0.3869 | 0.388 / 0.385 | 0.389 / 0.389 | 0.402 / 0.398 | 0.387 / 0.400 |

## ETTm2 (test MSE / MAE) — unchanged from 11-24

| H | first chain | extension | SpikF | SpikeRNN | iSpikformer | PatchTST |
|---|---|---|---|---|---|---|
| 96 | 0.1748 / 0.2525 | 0.1745 / 0.2521 | 0.175 / 0.251 | 0.176 / 0.251 | 0.178 / 0.256 | 0.175 / 0.259 |
| 192 | 0.2405 / 0.2965 | 0.2408 / 0.2965 | 0.242 / 0.296 | 0.243 / 0.296 | 0.243 / 0.298 | 0.241 / 0.302 |
| 336 | 0.3019 / 0.3360 | 0.3007 / 0.3351 | 0.302 / 0.336 | 0.302 / 0.335 | 0.304 / 0.338 | 0.305 / 0.343 |
| 720 | 0.4018 / 0.3940 | 0.3999 / 0.3937 | 0.405 / 0.397 | 0.409 / 0.396 | 0.406 / 0.398 | 0.402 / 0.400 |
| Avg | 0.2797 / 0.3197 | 0.2790 / 0.3194 | 0.281 / 0.320 | 0.283 / 0.319 | 0.283 / 0.322 | 0.281 / 0.326 |

## Judgment (16-cell means; margin = ref − ours, ≥ +0.0010 needed for MSE and MAE)

### first chain: 16-cell MSE / MAE **0.3716 / 0.3815** (11-24 full table: 0.3711 / 0.3821)

| reference | role | ref 16-cell MSE / MAE | margin MSE / MAE | passes | cells with lower MSE / MAE (of 16) |
|---|---|---|---|---|---|
| SpikF | SNN | 0.3700 / 0.3819 | -0.0016 / +0.0004 | no | 7 / 7 |
| SpikeRNN | SNN | 0.3761 / 0.3845 | +0.0044 / +0.0030 | yes | 13 / 12 |
| iSpikformer | SNN | 0.3875 / 0.3928 | +0.0159 / +0.0112 | yes | 16 / 16 |
| Informer | DNN | 1.9607 / 1.0169 | +1.5891 / +0.6354 | yes | 16 / 16 |
| FEDformer | DNN | 0.4074 / 0.4276 | +0.0358 / +0.0460 | yes | 13 / 16 |
| PatchTST | DNN | 0.3809 / 0.3969 | +0.0093 / +0.0154 | yes | 13 / 16 |
| ETSformer | DNN | 0.4256 / 0.4320 | +0.0540 / +0.0505 | yes | 16 / 16 |
| Autoformer | DNN | 0.4651 / 0.4586 | +0.0934 / +0.0771 | yes | 16 / 16 |

Energy (syops): below SpikF in **16 / 16** cells (needed 16); mean of cell ratios ours / SpikF 0.0118, largest 0.0209 (ETTm2 H720); 16-cell mean 11.11 µJ (SpikF 907.98 µJ).
Accuracy rows (3 SNN + 5 DNN): NOT all pass: SpikF. **Verdict: NOT PASSED**

### extension: 16-cell MSE / MAE **0.3724 / 0.3826** (11-24 full table: 0.3729 / 0.3829)

| reference | role | ref 16-cell MSE / MAE | margin MSE / MAE | passes | cells with lower MSE / MAE (of 16) |
|---|---|---|---|---|---|
| SpikF | SNN | 0.3700 / 0.3819 | -0.0024 / -0.0007 | no | 9 / 6 |
| SpikeRNN | SNN | 0.3761 / 0.3845 | +0.0037 / +0.0019 | yes | 12 / 10 |
| iSpikformer | SNN | 0.3875 / 0.3928 | +0.0151 / +0.0102 | yes | 16 / 16 |
| Informer | DNN | 1.9607 / 1.0169 | +1.5883 / +0.6343 | yes | 16 / 16 |
| FEDformer | DNN | 0.4074 / 0.4276 | +0.0350 / +0.0450 | yes | 13 / 16 |
| PatchTST | DNN | 0.3809 / 0.3969 | +0.0085 / +0.0143 | yes | 13 / 16 |
| ETSformer | DNN | 0.4256 / 0.4320 | +0.0533 / +0.0494 | yes | 16 / 16 |
| Autoformer | DNN | 0.4651 / 0.4586 | +0.0927 / +0.0761 | yes | 16 / 16 |

Energy (syops): below SpikF in **16 / 16** cells (needed 16); mean of cell ratios ours / SpikF 0.0259, largest 0.0698 (ETTm2 H720); 16-cell mean 24.09 µJ (SpikF 907.98 µJ).
Accuracy rows (3 SNN + 5 DNN): NOT all pass: SpikF. **Verdict: NOT PASSED**

## Report: ETTh1 / ETTm1 against the 11-24 full table (settings copied from ETTh2 / ETTm2), test MSE seed mean

| model | cell | 11-24 | retuned | difference |
|---|---|--:|--:|--:|
| first chain | ETTh1 H96 | 0.3775 | 0.3833 | +0.0058 |
| first chain | ETTh1 H192 | 0.4382 | 0.4396 | +0.0014 |
| first chain | ETTh1 H336 | 0.4818 | 0.4820 | +0.0002 |
| first chain | ETTh1 H720 | 0.4925 | 0.4788 | -0.0136 |
| first chain | ETTm1 H96 | 0.3171 | 0.3198 | +0.0027 |
| first chain | ETTm1 H192 | 0.3694 | 0.3730 | +0.0036 |
| first chain | ETTm1 H336 | 0.3980 | 0.4031 | +0.0051 |
| first chain | ETTm1 H720 | 0.4645 | 0.4676 | +0.0031 |
| extension | ETTh1 H96 | 0.3780 | 0.3764 | -0.0017 |
| extension | ETTh1 H192 | 0.4404 | 0.4416 | +0.0012 |
| extension | ETTh1 H336 | 0.4891 | 0.4842 | -0.0048 |
| extension | ETTh1 H720 | 0.4952 | 0.4924 | -0.0027 |
| extension | ETTm1 H96 | 0.3204 | 0.3189 | -0.0015 |
| extension | ETTm1 H192 | 0.3715 | 0.3724 | +0.0009 |
| extension | ETTm1 H336 | 0.4071 | 0.4077 | +0.0006 |
| extension | ETTm1 H720 | 0.4737 | 0.4732 | -0.0005 |

## Report: per dataset, mean of the 4 horizons against SpikF (+ = ours higher = worse)

| model | dataset | MSE ours − SpikF | MAE ours − SpikF |
|---|---|--:|--:|
| first chain | ETTh1 | +0.0064 | +0.0001 |
| first chain | ETTh2 | -0.0018 | -0.0024 |
| first chain | ETTm1 | +0.0031 | +0.0011 |
| first chain | ETTm2 | -0.0013 | -0.0003 |
| extension | ETTh1 | +0.0092 | +0.0053 |
| extension | ETTh2 | -0.0030 | -0.0036 |
| extension | ETTm1 | +0.0053 | +0.0017 |
| extension | ETTm2 | -0.0020 | -0.0006 |

## Energy per cell (syops µJ per input window; ours seed mean, SpikF one checkpoint)

| cell | first chain | extension | SpikF | first / SpikF | ext / SpikF |
|---|--:|--:|--:|--:|--:|
| ETTh1 H96 | 6.04 | 19.30 | 819.12 | 0.0074 | 0.0236 |
| ETTh1 H192 | 8.40 | 24.08 | 854.73 | 0.0098 | 0.0282 |
| ETTh1 H336 | 11.96 | 31.24 | 908.15 | 0.0132 | 0.0344 |
| ETTh1 H720 | 21.46 | 50.32 | 1050.59 | 0.0204 | 0.0479 |
| ETTh2 H96 | 5.92 | 4.50 | 819.00 | 0.0072 | 0.0055 |
| ETTh2 H192 | 8.27 | 6.07 | 854.61 | 0.0097 | 0.0071 |
| ETTh2 H336 | 11.84 | 8.45 | 908.03 | 0.0130 | 0.0093 |
| ETTh2 H720 | 21.31 | 14.76 | 1050.47 | 0.0203 | 0.0141 |
| ETTm1 H96 | 5.20 | 5.52 | 819.06 | 0.0063 | 0.0067 |
| ETTm1 H192 | 6.41 | 6.71 | 854.67 | 0.0075 | 0.0078 |
| ETTm1 H336 | 8.17 | 8.50 | 908.09 | 0.0090 | 0.0094 |
| ETTm1 H720 | 12.91 | 13.23 | 1050.53 | 0.0123 | 0.0126 |
| ETTm2 H96 | 6.58 | 32.44 | 818.66 | 0.0080 | 0.0396 |
| ETTm2 H192 | 8.94 | 38.78 | 854.26 | 0.0105 | 0.0454 |
| ETTm2 H336 | 12.49 | 48.25 | 907.68 | 0.0138 | 0.0532 |
| ETTm2 H720 | 21.94 | 73.32 | 1050.10 | 0.0209 | 0.0698 |

## Seed spread (std of test MSE over the 3 seeds, mean over the 16 cells)

- first chain: 0.0018
- extension: 0.0021
