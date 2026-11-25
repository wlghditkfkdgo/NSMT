# ETTh1 / ETTm1 retuning of the LIF SpkMixer, stage lr (LR); validation only

Rule: rt_plan.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; lowest wins; within 1e-12 of the current value keeps it). No test values are shown at this stage.

## shared: ETTh1 (before this stage: LR 0.0005, LAYERS 2, PATCH 8, EMB 32, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| LR=0.0001 | 0.548186 | 0.851302 | 0.999464 |  |
| LR=0.00025 | 0.546604 | 0.851465 | 0.998118 | **selected** |
| LR=0.0005 | 0.548560 | 0.851635 | 1.000000 | (current) |
| LR=0.001 | 0.548106 | 0.851250 | 0.999360 |  |
| LR=0.002 | 0.551588 | 0.851725 | 1.002813 |  |

## shared: ETTm1 (before this stage: LR 0.001, LAYERS 2, PATCH 8, EMB 64, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| LR=0.0001 | 0.406620 | 0.650941 | 1.002108 |  |
| LR=0.00025 | 0.404664 | 0.648092 | 0.997504 |  |
| LR=0.0005 | 0.403903 | 0.647476 | 0.996092 | **selected** |
| LR=0.001 | 0.404945 | 0.650889 | 1.000000 | (current) |
| LR=0.002 | 0.405236 | 0.649987 | 0.999666 |  |

Selected (shared): ETTh1 LR=0.00025; ETTm1 LR=0.0005

