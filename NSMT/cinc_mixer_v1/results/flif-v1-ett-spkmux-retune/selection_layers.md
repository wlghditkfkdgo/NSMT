# ETTh1 / ETTm1 retuning of the LIF SpkMixer, stage layers (LAYERS); validation only

Rule: rt_plan.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; lowest wins; within 1e-12 of the current value keeps it). No test values are shown at this stage.

## shared: ETTh1 (before this stage: LR 0.00025, LAYERS 2, PATCH 8, EMB 32, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| LAYERS=1 | 0.547396 | 0.849980 | 0.999852 | **selected** |
| LAYERS=2 | 0.546604 | 0.851465 | 1.000000 | (current) |
| LAYERS=3 | 0.547562 | 0.850871 | 1.000527 |  |

## shared: ETTm1 (before this stage: LR 0.0005, LAYERS 2, PATCH 8, EMB 64, MLPR 1, BLOCKS 1, WD 0.06)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| LAYERS=1 | 0.405535 | 0.650318 | 1.004214 |  |
| LAYERS=2 | 0.403903 | 0.647476 | 1.000000 | **selected** (current) |
| LAYERS=3 | 0.403977 | 0.649383 | 1.001564 |  |

Selected (shared): ETTh1 LAYERS=1; ETTm1 LAYERS=2

