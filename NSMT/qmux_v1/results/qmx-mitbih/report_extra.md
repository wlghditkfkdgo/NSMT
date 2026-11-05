# MIT-BIH stage: independent recomputation and report-only tables

The 32 pre-registered rows and the adoption table of summary.md were reproduced by separate code. All 54 runs: 30 epochs, lr 0.01, 32833 test samples.
Everything below is report only. Plain LIF and f-LIF controls come from this suite, plain population controls from flif-v1-mitbih-query.

## V1 by class: muxlin minus plain query, F1 %p (seed mean)

| neuron | residual | macro | N | S | V | F | Q |
|---|---|---:|---:|---:|---:|---:|---:|
| lif | OR | -0.01 | +0.01 | +0.53 | +0.04 | -0.73 | +0.11 |
| lif | IAND | +0.09 | +0.00 | +0.49 | +0.12 | -0.20 | +0.03 |
| flif | OR | -0.06 | +0.00 | +0.36 | -0.11 | -0.68 | +0.12 |
| flif | IAND | -0.01 | -0.02 | -0.15 | -0.28 | +0.25 | +0.14 |
| pop_lif | OR | +0.05 | +0.01 | -0.01 | +0.05 | +0.15 | +0.04 |
| pop_lif | IAND | -0.01 | +0.00 | -0.33 | +0.08 | +0.06 | +0.13 |
| pop_flif | OR | +0.09 | +0.02 | +0.12 | -0.09 | +0.32 | +0.09 |
| pop_flif | IAND | -0.01 | +0.02 | -0.13 | -0.05 | +0.03 | +0.06 |

## Firing rates, energy, seed range (mean over 3 seeds)

| condition | rho_neo | rho_hippo | energy µJ | lowest and highest seed macro-F1 % |
|---|---:|---:|---:|---|
| nsmt4q_lif | 0.0733 | 0.1383 | 10.38 | 91.53, 92.12 |
| qmx_lif_muxlin_or | 0.0687 | 0.1369 | 11.58 | 91.72, 91.95 |
| qmx_lif_muxlin_iand | 0.0698 | 0.1404 | 11.21 | 91.86, 92.02 |
| qmx_lif_or_or | 0.0604 | 0.1418 | 11.68 | 91.84, 92.77 |
| qmx_lif_or_iand | 0.0663 | 0.1433 | 10.11 | 92.03, 92.50 |
| nsmt4q_flif | 0.0673 | 0.1400 | 10.48 | 91.85, 92.53 |
| qmx_flif_muxlin_or | 0.0532 | 0.1373 | 11.60 | 91.94, 92.26 |
| qmx_flif_muxlin_iand | 0.0570 | 0.1445 | 11.40 | 91.84, 92.44 |
| qmx_flif_or_or | 0.0457 | 0.1419 | 11.43 | 91.83, 92.20 |
| qmx_flif_or_iand | 0.0499 | 0.1445 | 10.35 | 91.77, 92.36 |
| nsmt4q_pop_lif | 0.0424 | 0.1476 | 11.10 | 92.00, 92.33 |
| qmx_pop_lif_muxlin_or | 0.0643 | 0.1373 | 13.65 | 92.08, 92.50 |
| qmx_pop_lif_muxlin_iand | 0.0485 | 0.1433 | 13.35 | 92.04, 92.28 |
| qmx_pop_lif_or_or | 0.0428 | 0.1380 | 12.63 | 92.14, 92.52 |
| qmx_pop_lif_or_iand | 0.0423 | 0.1451 | 11.46 | 91.90, 92.27 |
| nsmt4q_pop_flif | 0.0623 | 0.1398 | 12.26 | 91.84, 92.18 |
| qmx_pop_flif_muxlin_or | 0.0569 | 0.1346 | 16.62 | 91.83, 92.29 |
| qmx_pop_flif_muxlin_iand | 0.0453 | 0.1409 | 16.27 | 91.82, 92.14 |
| qmx_pop_flif_or_or | 0.0510 | 0.1436 | 14.58 | 92.02, 92.49 |
| qmx_pop_flif_or_iand | 0.0424 | 0.1421 | 13.10 | 91.86, 92.19 |

## Highest mean macro-F1 (chosen after seeing the test values, so biased): qmx_lif_or_or 92.31

| comparison | mean %p | seed 42, 7, 2026 | verdict | energy A/B |
|---|---:|---|---|---:|
| vs nsmt4q_lif | +0.47 | +1.24, +0.20, -0.02 | no clear difference | 1.13 |
| vs Spikformer-style (default) | +0.08 | +0.35, +0.07, -0.17 | no clear difference | 1.22 |
| vs Spikformer-style (tuned) | +0.52 | +0.66, +0.58, +0.33 | better | 1.22 |
| vs NSMT rerun | +0.72 | +1.13, +0.75, +0.30 | better | 0.49 |

## The ETT lowest-MSE condition on MIT-BIH (also chosen after seeing test values): qmx_pop_flif_or_or 92.26

| comparison | mean %p | seed 42, 7, 2026 | verdict | energy A/B |
|---|---:|---|---|---:|
| vs nsmt4q_pop_flif | +0.25 | -0.01, +0.65, +0.10 | no clear difference | 1.19 |
| vs Spikformer-style (default) | +0.04 | -0.39, +0.25, +0.27 | no clear difference | 1.53 |
| vs Spikformer-style (tuned) | +0.48 | -0.08, +0.75, +0.77 | no clear difference | 1.52 |
| vs NSMT rerun | +0.68 | +0.38, +0.92, +0.74 | better | 0.61 |

Range of the mean macro-F1 over the 20 query conditions: 91.83 to 92.31 %.

## Reference: Hippocampus-only NSMT at its own setting (no Neocortex path, same training rule; macro-F1 91.83; NOT an ablation of the query model, which has T = 4)

| comparison | mean %p | seed 42, 7, 2026 | verdict | energy A/B |
|---|---:|---|---|---:|
| NSMT rerun vs hippo-only | -0.25 | -0.42, -0.25, -0.07 | no clear difference | 0.92 |
| nsmt4q_lif vs hippo-only | +0.01 | -0.53, +0.31, +0.25 | no clear difference | 0.40 |
| nsmt4q_flif vs hippo-only | +0.38 | +0.18, +0.71, +0.24 | no clear difference | 0.40 |
| nsmt4q_pop_lif vs hippo-only | +0.35 | +0.13, +0.19, +0.72 | no clear difference | 0.43 |
| nsmt4q_pop_flif vs hippo-only | +0.19 | -0.02, +0.02, +0.57 | no clear difference | 0.47 |
| qmx_lif_muxlin_or vs hippo-only | +0.00 | -0.24, -0.09, +0.34 | no clear difference | 0.45 |
| qmx_lif_muxlin_iand vs hippo-only | +0.10 | -0.16, +0.20, +0.25 | no clear difference | 0.43 |
| qmx_flif_muxlin_or vs hippo-only | +0.32 | -0.12, +0.45, +0.63 | no clear difference | 0.45 |
| qmx_flif_muxlin_iand vs hippo-only | +0.37 | -0.21, +0.49, +0.83 | no clear difference | 0.44 |
| qmx_pop_lif_muxlin_or vs hippo-only | +0.39 | +0.44, +0.26, +0.48 | no clear difference | 0.52 |
| qmx_pop_lif_muxlin_iand vs hippo-only | +0.34 | +0.10, +0.23, +0.68 | no clear difference | 0.51 |
| qmx_pop_flif_muxlin_or vs hippo-only | +0.28 | +0.15, +0.02, +0.68 | no clear difference | 0.64 |
| qmx_pop_flif_muxlin_iand vs hippo-only | +0.18 | -0.01, +0.32, +0.22 | no clear difference | 0.63 |
