# ETTh1 / ETTm1 retuning of the LIF SpkMixer, stage mlp (MLPR); validation only

Rule: rt_plan.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; lowest wins; within 1e-12 of the current value keeps it). No test values are shown at this stage.

## first chain (patch <= 16, wd <= 0.1): ETTh1 (before this stage: LR 0.00025, LAYERS 1, PATCH 8, EMB 32, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| MLPR=0.5 | 0.547263 | 0.851067 | 1.000518 |  |
| MLPR=1 | 0.547396 | 0.849980 | 1.000000 | (current) |
| MLPR=2 | 0.546341 | 0.851345 | 0.999839 |  |
| MLPR=4 | 0.547116 | 0.849369 | 0.999384 | **selected** |

## first chain (patch <= 16, wd <= 0.1): ETTm1 (before this stage: LR 0.0005, LAYERS 2, PATCH 16, EMB 32, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| MLPR=0.5 | 0.398611 | 0.647141 | 0.998524 | **selected** |
| MLPR=1 | 0.399605 | 0.647442 | 1.000000 | (current) |
| MLPR=2 | 0.399871 | 0.647761 | 1.000580 |  |
| MLPR=4 | 0.401089 | 0.647693 | 1.002051 |  |

Selected (first chain (patch <= 16, wd <= 0.1)): ETTh1 MLPR=4; ETTm1 MLPR=0.5

## extension (patch <= 32, wd <= 0.06): ETTh1 (before this stage: LR 0.00025, LAYERS 1, PATCH 32, EMB 256, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| MLPR=0.5 | 0.544921 | 0.850381 | 1.000510 |  |
| MLPR=1 | 0.544676 | 0.849895 | 1.000000 | **selected** (current) |
| MLPR=2 | 0.545166 | 0.850584 | 1.000855 |  |
| MLPR=4 | 0.545751 | 0.850525 | 1.001358 |  |

## extension (patch <= 32, wd <= 0.06): ETTm1 (before this stage: LR 0.0005, LAYERS 2, PATCH 32, EMB 64, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| MLPR=0.5 | 0.399850 | 0.648308 | 1.001070 |  |
| MLPR=1 | 0.399432 | 0.647600 | 1.000000 | **selected** (current) |
| MLPR=2 | 0.399552 | 0.648398 | 1.000767 |  |
| MLPR=4 | 0.399639 | 0.648546 | 1.000990 |  |

Selected (extension (patch <= 32, wd <= 0.06)): ETTh1 MLPR=1; ETTm1 MLPR=1

