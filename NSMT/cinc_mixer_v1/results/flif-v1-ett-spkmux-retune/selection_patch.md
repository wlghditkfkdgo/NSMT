# ETTh1 / ETTm1 retuning of the LIF SpkMixer, stage patch (PATCH); validation only

Rule: rt_plan.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; lowest wins; within 1e-12 of the current value keeps it). No test values are shown at this stage.

## first chain (patch <= 16, wd <= 0.1): ETTh1 (before this stage: LR 0.00025, LAYERS 1, PATCH 8, EMB 32, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| PATCH=4 | 0.549909 | 0.848272 | 1.001291 |  |
| PATCH=8 | 0.547396 | 0.849980 | 1.000000 | **selected** (current) |
| PATCH=16 | 0.548120 | 0.851050 | 1.001290 |  |

## first chain (patch <= 16, wd <= 0.1): ETTm1 (before this stage: LR 0.0005, LAYERS 2, PATCH 8, EMB 64, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| PATCH=4 | 0.410824 | 0.654761 | 1.014193 |  |
| PATCH=8 | 0.403903 | 0.647476 | 1.000000 | (current) |
| PATCH=16 | 0.400828 | 0.646067 | 0.995105 | **selected** |

Selected (first chain (patch <= 16, wd <= 0.1)): ETTh1 PATCH=8; ETTm1 PATCH=16

## extension (patch <= 32, wd <= 0.06): ETTh1 (before this stage: LR 0.00025, LAYERS 1, PATCH 8, EMB 32, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| PATCH=4 | 0.549909 | 0.848272 | 1.001291 |  |
| PATCH=8 | 0.547396 | 0.849980 | 1.000000 | (current) |
| PATCH=16 | 0.548120 | 0.851050 | 1.001290 |  |
| PATCH=24 | 0.549011 | 0.851217 | 1.002203 |  |
| PATCH=32 | 0.547066 | 0.850331 | 0.999905 | **selected** |

## extension (patch <= 32, wd <= 0.06): ETTm1 (before this stage: LR 0.0005, LAYERS 2, PATCH 8, EMB 64, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| PATCH=4 | 0.410824 | 0.654761 | 1.014193 |  |
| PATCH=8 | 0.403903 | 0.647476 | 1.000000 | (current) |
| PATCH=16 | 0.400828 | 0.646067 | 0.995105 |  |
| PATCH=24 | 0.399605 | 0.648375 | 0.995373 |  |
| PATCH=32 | 0.399432 | 0.647600 | 0.994560 | **selected** |

Selected (extension (patch <= 32, wd <= 0.06)): ETTh1 PATCH=32; ETTm1 PATCH=32

