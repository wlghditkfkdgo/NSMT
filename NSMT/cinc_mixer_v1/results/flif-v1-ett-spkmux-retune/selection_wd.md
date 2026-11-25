# ETTh1 / ETTm1 retuning of the LIF SpkMixer, stage wd (WD); validation only

Rule: rt_plan.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; lowest wins; within 1e-12 of the current value keeps it). No test values are shown at this stage.

## first chain (patch <= 16, wd <= 0.1): ETTh1 (before this stage: LR 0.00025, LAYERS 1, PATCH 8, EMB 32, MLPR 4, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| WD=0 | 0.547812 | 0.850021 | 1.001020 |  |
| WD=0.01 | 0.547575 | 0.849937 | 1.000754 |  |
| WD=0.06 | 0.547116 | 0.849369 | 1.000000 | (current) |
| WD=0.1 | 0.546212 | 0.850252 | 0.999694 | **selected** |

## first chain (patch <= 16, wd <= 0.1): ETTm1 (before this stage: LR 0.0005, LAYERS 2, PATCH 16, EMB 32, MLPR 0.5, BLOCKS 3, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| WD=0 | 0.399096 | 0.647632 | 1.001918 |  |
| WD=0.01 | 0.399411 | 0.647118 | 1.001915 |  |
| WD=0.06 | 0.398117 | 0.646743 | 1.000000 | **selected** (current) |
| WD=0.1 | 0.398860 | 0.646695 | 1.000897 |  |

Selected (first chain (patch <= 16, wd <= 0.1)): ETTh1 WD=0.1; ETTm1 WD=0.06

## extension (patch <= 32, wd <= 0.06): ETTh1 (before this stage: LR 0.00025, LAYERS 1, PATCH 32, EMB 256, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| WD=0 | 0.545264 | 0.850218 | 1.000729 |  |
| WD=0.005 | 0.545365 | 0.850129 | 1.000770 |  |
| WD=0.01 | 0.545698 | 0.850350 | 1.001205 |  |
| WD=0.02 | 0.545135 | 0.850245 | 1.000627 |  |
| WD=0.05 | 0.545205 | 0.849901 | 1.000489 |  |
| WD=0.06 | 0.544676 | 0.849895 | 1.000000 | **selected** (current) |

## extension (patch <= 32, wd <= 0.06): ETTm1 (before this stage: LR 0.0005, LAYERS 2, PATCH 32, EMB 64, MLPR 1, BLOCKS 2, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| WD=0 | 0.400342 | 0.649203 | 1.002771 |  |
| WD=0.005 | 0.399876 | 0.648576 | 1.001703 |  |
| WD=0.01 | 0.399883 | 0.649851 | 1.002694 |  |
| WD=0.02 | 0.399906 | 0.649365 | 1.002348 |  |
| WD=0.05 | 0.399968 | 0.648785 | 1.001980 |  |
| WD=0.06 | 0.398448 | 0.648692 | 1.000000 | **selected** (current) |

Selected (extension (patch <= 32, wd <= 0.06)): ETTh1 WD=0.06; ETTm1 WD=0.06

