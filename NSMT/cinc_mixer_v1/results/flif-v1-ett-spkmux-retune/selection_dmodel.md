# ETTh1 / ETTm1 retuning of the LIF SpkMixer, stage dmodel (EMB); validation only

Rule: rt_plan.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; lowest wins; within 1e-12 of the current value keeps it). No test values are shown at this stage.

## first chain (patch <= 16, wd <= 0.1): ETTh1 (before this stage: LR 0.00025, LAYERS 1, PATCH 8, EMB 32, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| EMB=32 | 0.547396 | 0.849980 | 1.000000 | **selected** (current) |
| EMB=64 | 0.549087 | 0.850453 | 1.001823 |  |
| EMB=128 | 0.551379 | 0.852126 | 1.004900 |  |
| EMB=256 | 0.555527 | 0.854952 | 1.010351 |  |

## first chain (patch <= 16, wd <= 0.1): ETTm1 (before this stage: LR 0.0005, LAYERS 2, PATCH 16, EMB 64, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| EMB=32 | 0.399605 | 0.647442 | 0.999538 | **selected** |
| EMB=64 | 0.400828 | 0.646067 | 1.000000 | (current) |
| EMB=128 | 0.404110 | 0.648916 | 1.006299 |  |
| EMB=256 | 0.408297 | 0.651252 | 1.013329 |  |

Selected (first chain (patch <= 16, wd <= 0.1)): ETTh1 EMB=32; ETTm1 EMB=32

## extension (patch <= 32, wd <= 0.06): ETTh1 (before this stage: LR 0.00025, LAYERS 1, PATCH 32, EMB 32, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| EMB=32 | 0.547066 | 0.850331 | 1.000000 | (current) |
| EMB=64 | 0.546314 | 0.849686 | 0.998933 |  |
| EMB=128 | 0.544858 | 0.849796 | 0.997668 |  |
| EMB=256 | 0.544676 | 0.849895 | 0.997560 | **selected** |

## extension (patch <= 32, wd <= 0.06): ETTm1 (before this stage: LR 0.0005, LAYERS 2, PATCH 32, EMB 64, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| EMB=32 | 0.400165 | 0.649180 | 1.002137 |  |
| EMB=64 | 0.399432 | 0.647600 | 1.000000 | **selected** (current) |
| EMB=128 | 0.400036 | 0.648320 | 1.001312 |  |
| EMB=256 | 0.401328 | 0.649931 | 1.004173 |  |

Selected (extension (patch <= 32, wd <= 0.06)): ETTh1 EMB=256; ETTm1 EMB=64

