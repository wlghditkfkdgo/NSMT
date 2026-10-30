# ETT check of muxlin `flif-v1-ett-spkmux` (LIF, residual OR; ETTh2 + ETTm2 x H96 / H720, seeds (42, 7, 2026))

Test MSE / MAE on the scaled data, best-validation weights. Energy = forecast-head-corrected total µJ per window (7 channels).

## Test MSE (mean ± sd over seeds; n < 3 marked)

| condition | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---:|---:|---:|---:|
| Spikformer-style | 0.2903 ± 0.0010 | 0.4223 ± 0.0019 | 0.1775 ± 0.0006 | 0.4034 ± 0.0013 |
| sdlin (L) | 0.2913 ± 0.0013 | 0.4265 ± 0.0134 | 0.1834 ± 0.0017 | 0.4133 ± 0.0026 |
| muxlin (s_d ? L : F ^ R) | 0.2930 ± 0.0016 | 0.4214 ± 0.0073 | 0.1826 ± 0.0001 | 0.4107 ± 0.0016 |

## Test MAE (mean ± sd over seeds; n < 3 marked)

| condition | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---:|---:|---:|---:|
| Spikformer-style | 0.3350 ± 0.0014 | 0.4366 ± 0.0015 | 0.2536 ± 0.0004 | 0.3959 ± 0.0004 |
| sdlin (L) | 0.3364 ± 0.0008 | 0.4415 ± 0.0085 | 0.2610 ± 0.0005 | 0.4023 ± 0.0007 |
| muxlin (s_d ? L : F ^ R) | 0.3374 ± 0.0009 | 0.4387 ± 0.0052 | 0.2605 ± 0.0002 | 0.4007 ± 0.0003 |

## Energy (corrected µJ per window), epochs run, parameters

| condition | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | epochs run | params H96 / H720 |
|---|---:|---:|---:|---:|---:|---|
| Spikformer-style | 9.30 | 10.79 | 10.28 | 11.48 | 15.4 | [169656, 174192] / [1128120, 1132656] |
| sdlin (L) | 9.10 | 11.88 | 11.87 | 14.45 | 11.2 | [169280, 181888] / [1127744, 1140352] |
| muxlin (s_d ? L : F ^ R) | 9.06 | 11.56 | 11.82 | 14.41 | 11.2 | [169280, 181888] / [1127744, 1140352] |

## Pre-registered comparisons (test MSE; r = 100 (A − B) / B per seed, %)

| question | cells better / none / worse | R | seed pairs A < B | verdict | energy A/B | criterion (2) | R on MAE (report) |
|---|---|---:|---|---|---:|---|---:|
| E1 muxlin vs Spikformer-style (criterion 2) | 0 / 2 / 2 | +1.36 | 3/12 | worse | 1.11 | not met | +1.29 |
| E2 muxlin vs sdlin | 1 / 3 / 0 | -0.39 | 8/12 | no clear difference | 0.99 |  | -0.22 |
| E3 sdlin vs Spikformer-style (criterion 2) | 0 / 2 / 2 | +1.77 | 3/12 | worse | 1.12 | not met | +1.52 |

## Per-cell mean r (%), per seed (42, 7, 2026), cell verdict (b better, w worse, . no clear difference)

| A − B | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| spkmx_lif_muxlin_or − spk_ssa | +0.93 (-0.13, +1.70, +1.22) . | -0.22 (+2.26, -0.62, -2.30) . | +2.91 (+2.45, +3.14, +3.13) w | +1.81 (+1.64, +1.64, +2.15) w |
| spkmx_lif_muxlin_or − spkmx_lif_sdlin_or | +0.61 (-0.43, +1.96, +0.31) . | -1.15 (-3.09, -0.44, +0.08) b | -0.42 (+0.61, -1.58, -0.27) . | -0.61 (-0.41, -1.19, -0.22) . |
| spkmx_lif_sdlin_or − spk_ssa | +0.32 (+0.30, -0.25, +0.91) . | +0.99 (+5.52, -0.18, -2.38) . | +3.35 (+1.84, +4.80, +3.41) w | +2.43 (+2.06, +2.87, +2.38) w |

**Decision rule:** E1 worse -> fold 1-4 and tuning of muxlin ON HOLD (user rule 10-31).

RESULT rows (new): 24; SKIP lines: 0; baseline rows: 12.
