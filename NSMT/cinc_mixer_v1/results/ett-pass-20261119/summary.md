# ETT pass criteria (user 11-19): current models, input 96, no training

Rules and sources: module docstring of pass_ett.py. Ours: test values of the best-validation weights, seeds 42 / 7 / 2026; energy = logged total_fft_uJ (syops as published + f-LIF-paper neuron energy).

## Transcription check

- verify: 60 checks (each a cell MSE / MAE pair), 0 mismatches

## Reference rows (look-back 96; MSE / MAE per cell: ETTh2 H96, ETTh2 H720, ETTm2 H96, ETTm2 H720)

| row | kind | source | MSE | MAE | 4-cell mean MSE / MAE |
|---|---|---|---|---|---|
| SpikF | SNN | SpikF Table 12 = NSMT Table 14 | 0.290, 0.416, 0.175, 0.405 | 0.336, 0.436, 0.251, 0.397 | 0.3215 / 0.3550 |
| iTransformer | DNN | iTransformer Table 10 | 0.297, 0.427, 0.180, 0.412 | 0.349, 0.445, 0.264, 0.407 | 0.3290 / 0.3663 |
| iTransformer, MAE loss | DNN | SpikF Table 9, 10 | 0.294, 0.423, 0.182, 0.411 | 0.342, 0.439, 0.265, 0.403 | 0.3275 / 0.3623 |
| RLinear | DNN | iTransformer Table 10 | 0.288, 0.420, 0.182, 0.407 | 0.338, 0.440, 0.265, 0.398 | 0.3242 / 0.3603 |
| PatchTST (re-run) | DNN | iTransformer Table 10 | 0.302, 0.431, 0.175, 0.402 | 0.348, 0.446, 0.259, 0.400 | 0.3275 / 0.3632 |
| PatchTST (other re-run) | DNN | TimeMixer Table 13 | 0.308, 0.436, 0.183, 0.412 | 0.355, 0.450, 0.270, 0.404 | 0.3347 / 0.3698 |
| TimeMixer | DNN | TimeMixer Table 13 | 0.289, 0.412, 0.175, 0.391 | 0.341, 0.434, 0.258, 0.396 | 0.3167 / 0.3572 |
| TimesNet | DNN | iTransformer Table 10 | 0.340, 0.462, 0.187, 0.408 | 0.374, 0.468, 0.267, 0.403 | 0.3493 / 0.3780 |
| FITS | DNN | SpikF Table 12 | 0.295, 0.431, 0.183, 0.407 | 0.350, 0.446, 0.266, 0.399 | 0.3290 / 0.3653 |
| DLinear (re-run) | DNN | iTransformer Table 10 | 0.333, 0.831, 0.193, 0.554 | 0.387, 0.657, 0.292, 0.522 | 0.4778 / 0.4645 |

## Criteria

### LIF SpkMixer (tuned 11-01: muxlin, IAND, 1 mixer layer per direction)

4-cell mean MSE / MAE: **0.3245 / 0.3567**; energy A/B vs SF-de (syops) 0.9825

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | -0.0030 / -0.0017 | no |
| SF-de (ours) | 0.3238 / 0.3557 | -0.0007 / -0.0010 | no |
| energy A/B vs SF-de (syops; bound 1.05) | 0.9825 | | yes |
| iTransformer | 0.3290 / 0.3663 | +0.0045 / +0.0096 | yes |
| iTransformer, MAE loss | 0.3275 / 0.3623 | +0.0030 / +0.0056 | yes |
| RLinear | 0.3242 / 0.3603 | -0.0002 / +0.0036 | no |
| PatchTST (re-run) | 0.3275 / 0.3632 | +0.0030 / +0.0066 | yes |
| PatchTST (other re-run) | 0.3347 / 0.3698 | +0.0103 / +0.0131 | yes |
| TimeMixer | 0.3167 / 0.3572 | -0.0077 / +0.0006 | no |
| TimesNet | 0.3493 / 0.3780 | +0.0248 / +0.0213 | yes |
| FITS | 0.3290 / 0.3653 | +0.0045 / +0.0086 | yes |
| DLinear (re-run) | 0.4778 / 0.4645 | +0.1533 / +0.1078 | yes |

- (1) SNN: **fail**; (2) Spikformer: **fail**; (3) DNN (all 9 rows): **fail** (7/9 rows); overall: **NOT PASSED**
- Needed for all three: 4-cell mean MSE ≤ 0.3157 (now 0.3245, short by 0.0087) and MAE ≤ 0.3540 (now 0.3567, short by 0.0027); plus energy A/B ≤ 1.05.
- Report (not a criterion): NSMT 0.3177 / 0.3518, ref − ours -0.0067 / -0.0049.

### SF-de (report: the same criteria applied to SF-de against itself is trivially "no" for (2))

4-cell mean MSE / MAE: **0.3238 / 0.3557**; energy A/B vs SF-de (syops) 1.0000

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | -0.0023 / -0.0007 | no |
| SF-de (ours) | 0.3238 / 0.3557 | +0.0000 / +0.0000 | no |
| energy A/B vs SF-de (syops; bound 1.05) | 1.0000 | | yes |
| iTransformer | 0.3290 / 0.3663 | +0.0052 / +0.0105 | yes |
| iTransformer, MAE loss | 0.3275 / 0.3623 | +0.0037 / +0.0065 | yes |
| RLinear | 0.3242 / 0.3603 | +0.0004 / +0.0045 | no |
| PatchTST (re-run) | 0.3275 / 0.3632 | +0.0037 / +0.0075 | yes |
| PatchTST (other re-run) | 0.3347 / 0.3698 | +0.0109 / +0.0140 | yes |
| TimeMixer | 0.3167 / 0.3572 | -0.0071 / +0.0015 | no |
| TimesNet | 0.3493 / 0.3780 | +0.0254 / +0.0223 | yes |
| FITS | 0.3290 / 0.3653 | +0.0052 / +0.0095 | yes |
| DLinear (re-run) | 0.4778 / 0.4645 | +0.1539 / +0.1088 | yes |

- (1) SNN: **fail**; (2) Spikformer: **fail**; (3) DNN (all 9 rows): **fail** (7/9 rows); overall: **NOT PASSED**
- Needed for all three: 4-cell mean MSE ≤ 0.3157 (now 0.3238, short by 0.0081) and MAE ≤ 0.3540 (now 0.3557, short by 0.0017); plus energy A/B ≤ 1.05.
- Report (not a criterion): NSMT 0.3177 / 0.3518, ref − ours -0.0061 / -0.0040.

