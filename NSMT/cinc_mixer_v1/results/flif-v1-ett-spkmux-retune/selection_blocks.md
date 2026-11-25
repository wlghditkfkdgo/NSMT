# ETTh1 / ETTm1 retuning of the LIF SpkMixer, stage blocks (BLOCKS); validation only

Rule: rt_plan.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; lowest wins; within 1e-12 of the current value keeps it). No test values are shown at this stage.

## first chain (patch <= 16, wd <= 0.1): ETTh1 (before this stage: LR 0.00025, LAYERS 1, PATCH 8, EMB 32, MLPR 4, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| BLOCKS=1 | 0.547116 | 0.849369 | 1.000000 | **selected** (current) |
| BLOCKS=2 | 0.546904 | 0.851549 | 1.001090 |  |
| BLOCKS=3 | 0.546360 | 0.851575 | 1.000608 |  |

## first chain (patch <= 16, wd <= 0.1): ETTm1 (before this stage: LR 0.0005, LAYERS 2, PATCH 16, EMB 32, MLPR 0.5, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| BLOCKS=1 | 0.398611 | 0.647141 | 1.000000 | (current) |
| BLOCKS=2 | 0.400117 | 0.647018 | 1.001794 |  |
| BLOCKS=3 | 0.398117 | 0.646743 | 0.999072 | **selected** |

Selected (first chain (patch <= 16, wd <= 0.1)): ETTh1 BLOCKS=1; ETTm1 BLOCKS=3

## extension (patch <= 32, wd <= 0.06): ETTh1 (before this stage: LR 0.00025, LAYERS 1, PATCH 32, EMB 256, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| BLOCKS=1 | 0.544676 | 0.849895 | 1.000000 | **selected** (current) |
| BLOCKS=2 | 0.545264 | 0.851191 | 1.001302 |  |
| BLOCKS=3 | 0.545227 | 0.850920 | 1.001109 |  |

## extension (patch <= 32, wd <= 0.06): ETTm1 (before this stage: LR 0.0005, LAYERS 2, PATCH 32, EMB 64, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| BLOCKS=1 | 0.399432 | 0.647600 | 1.000000 | (current) |
| BLOCKS=2 | 0.398448 | 0.648692 | 0.999611 | **selected** |
| BLOCKS=3 | 0.399686 | 0.649305 | 1.001634 |  |

Selected (extension (patch <= 32, wd <= 0.06)): ETTh1 BLOCKS=1; ETTm1 BLOCKS=2

