# ETT stage: independent recomputation and report-only tables

Recount: 204 rows, largest |recomputed - logged| MSE 0.00e+00, head input binary in all. The 32 pre-registered rows of summary.md were reproduced by separate code.
Everything below is report only. Best epoch counts from 0.

## Best validation epoch (median over runs; share of runs with best epoch <= 1)

| Neocortex | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---:|---:|---:|---:|
| plain query | 3 (50 %, n = 12) | 4 (0 %, n = 12) | 7 (0 %, n = 12) | 0 (100 %, n = 12) |
| bidirectional muxlin | 1 (54 %, n = 24) | 3 (4 %, n = 24) | 7 (0 %, n = 24) | 0 (100 %, n = 24) |
| bidirectional OR | 1 (58 %, n = 24) | 3 (4 %, n = 24) | 7 (0 %, n = 24) | 1 (100 %, n = 24) |

## Seed spread: standard deviation over the 3 seeds as % of the mean MSE

| condition | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---:|---:|---:|---:|
| nsmt4q_lif | 0.82 | 0.90 | 0.23 | 0.64 |
| nsmt4q_flif | 1.29 | 0.87 | 0.56 | 0.72 |
| nsmt4q_pop_lif | 1.53 | 0.34 | 0.09 | 0.54 |
| nsmt4q_pop_flif | 1.71 | 0.45 | 0.54 | 0.35 |
| qmx_lif_muxlin_or | 0.46 | 0.82 | 0.08 | 0.30 |
| qmx_flif_muxlin_or | 0.65 | 1.06 | 0.40 | 0.33 |
| qmx_pop_lif_muxlin_or | 1.09 | 1.14 | 0.22 | 0.14 |
| qmx_pop_flif_muxlin_or | 2.24 | 0.44 | 0.32 | 0.24 |

## Firing rates and energy (mean over 12 runs)

| condition | rho_neo | rho_hippo | energy µJ |
|---|---:|---:|---:|
| nsmt4q_lif | 0.0535 | 0.0565 | 11.40 |
| qmx_lif_muxlin_or | 0.0538 | 0.0567 | 13.33 |
| qmx_lif_muxlin_iand | 0.0579 | 0.0572 | 13.18 |
| qmx_lif_or_or | 0.0421 | 0.0577 | 12.30 |
| qmx_lif_or_iand | 0.0515 | 0.0564 | 12.00 |
| nsmt4q_flif | 0.0472 | 0.0570 | 11.76 |
| qmx_flif_muxlin_or | 0.0476 | 0.0566 | 14.68 |
| qmx_flif_muxlin_iand | 0.0506 | 0.0577 | 14.59 |
| qmx_flif_or_or | 0.0359 | 0.0573 | 13.06 |
| qmx_flif_or_iand | 0.0464 | 0.0585 | 12.78 |
| nsmt4q_pop_lif | 0.0190 | 0.0531 | 12.94 |
| qmx_pop_lif_muxlin_or | 0.0395 | 0.0534 | 18.96 |
| qmx_pop_lif_muxlin_iand | 0.0412 | 0.0532 | 18.91 |
| qmx_pop_lif_or_or | 0.0174 | 0.0547 | 15.31 |
| qmx_pop_lif_or_iand | 0.0209 | 0.0555 | 15.18 |
| nsmt4q_pop_flif | 0.0196 | 0.0541 | 15.74 |
| qmx_pop_flif_muxlin_or | 0.0345 | 0.0547 | 29.07 |
| qmx_pop_flif_muxlin_iand | 0.0322 | 0.0538 | 28.95 |
| qmx_pop_flif_or_or | 0.0167 | 0.0562 | 20.94 |
| qmx_pop_flif_or_iand | 0.0167 | 0.0547 | 20.77 |

## Validation only: mean over cells of best validation L1 relative to the plain query of the same neuron, %

| neuron | muxlin OR | muxlin IAND | bidirectional OR, residual OR | bidirectional OR, residual IAND |
|---|---:|---:|---:|---:|
| lif | -0.00 | -0.00 | +0.11 | -0.02 |
| flif | +0.06 | -0.04 | -0.04 | -0.08 |
| pop_lif | -0.12 | -0.29 | -0.01 | -0.28 |
| pop_flif | +0.05 | -0.04 | +0.13 | -0.12 |

## Lowest mean test MSE (chosen after seeing the test values, so biased): qmx_pop_flif_or_or 0.3196; next nsmt4q_pop_flif 0.3203

| comparison | R % | lower pairs | verdict | energy A/B | per cell r % |
|---|---:|---:|---|---:|---|
| vs nsmt4q_pop_flif | -0.17 | 7/12 | no clear difference | 1.34 | -0.46, -0.29, +0.14, -0.06 |
| vs Spikformer-style (tuned) | -0.98 | 9/12 | no clear difference | 2.00 | -0.91, -2.36, +0.04, -0.67 |
| vs NSMT rerun | +0.02 | 5/12 | no clear difference | 0.32 | -0.64, -0.71, +1.37, +0.06 |

## Reference: Hippocampus-only NSMT, T = 24 (no Neocortex path, same training rule; mean test MSE 0.3179; NOT an ablation of the query model, which has T = 4; its energy is the logged value)

| comparison | R % | lower pairs | verdict | energy A/B | per cell r % |
|---|---:|---:|---|---:|---|
| NSMT rerun vs hippo-only | +0.69 | 2/12 | no clear difference | 0.99 | +1.42, +1.00, +0.26, +0.07 |
| nsmt4q_lif vs hippo-only | +1.67 | 1/12 | worse | 0.17 | +2.46, +1.59, +1.94, +0.68 |
| nsmt4q_flif vs hippo-only | +1.62 | 1/12 | worse | 0.18 | +2.63, +1.30, +1.79, +0.76 |
| nsmt4q_pop_lif vs hippo-only | +1.05 | 1/12 | worse | 0.20 | +1.99, +0.58, +1.21, +0.41 |
| nsmt4q_pop_flif vs hippo-only | +0.88 | 2/12 | no clear difference | 0.24 | +1.24, +0.57, +1.49, +0.20 |
| qmx_lif_muxlin_or vs hippo-only | +1.61 | 0/12 | worse | 0.20 | +2.04, +1.60, +1.84, +0.97 |
| qmx_lif_muxlin_iand vs hippo-only | +1.38 | 1/12 | worse | 0.20 | +1.20, +1.56, +1.95, +0.80 |
| qmx_flif_muxlin_or vs hippo-only | +1.27 | 1/12 | worse | 0.22 | +1.09, +1.47, +1.87, +0.64 |
| qmx_flif_muxlin_iand vs hippo-only | +1.48 | 2/12 | worse | 0.22 | +1.98, +1.72, +1.44, +0.79 |
| qmx_pop_lif_muxlin_or vs hippo-only | +1.26 | 2/12 | worse | 0.29 | +2.35, +1.09, +1.44, +0.17 |
| qmx_pop_lif_muxlin_iand vs hippo-only | +1.27 | 1/12 | worse | 0.29 | +2.01, +1.31, +1.32, +0.44 |
| qmx_pop_flif_muxlin_or vs hippo-only | +1.09 | 3/12 | worse | 0.44 | +2.36, +0.00, +1.69, +0.30 |
| qmx_pop_flif_muxlin_iand vs hippo-only | +1.14 | 1/12 | worse | 0.44 | +2.22, +0.66, +1.38, +0.31 |

## How close the V1 rows came to the rule (better needs R <= -1 % and at least 8 of 12 pairs lower)

| neuron | residual | R % | lower pairs | verdict |
|---|---|---:|---:|---|
| lif | OR | -0.05 | 7/12 | no clear difference |
| lif | IAND | -0.27 | 6/12 | no clear difference |
| flif | OR | -0.34 | 6/12 | no clear difference |
| flif | IAND | -0.13 | 5/12 | no clear difference |
| pop_lif | OR | +0.21 | 4/12 | no clear difference |
| pop_lif | IAND | +0.22 | 6/12 | no clear difference |
| pop_flif | OR | +0.21 | 5/12 | no clear difference |
| pop_flif | IAND | +0.26 | 4/12 | no clear difference |
