# ETT tuning of the LIF SpkMixer, stage blocks (BLOCKS); validation only

Rule: summarize_ettune.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; lowest wins; ties keep the current value). No test values are shown at this stage.

## ETTh2 (before this stage: PATCH=24)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| BLOCKS=1 | 0.311171 | 0.535842 | 1.000000 | **selected** (current) |
| BLOCKS=2 | 0.311440 | 0.536414 | 1.000966 |  |
| BLOCKS=3 | 0.311627 | 0.535945 | 1.000830 |  |

## ETTm2 (before this stage: PATCH=24, EMB=256)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| BLOCKS=1 | 0.235556 | 0.360480 | 1.000000 | (current) |
| BLOCKS=2 | 0.235577 | 0.360274 | 0.999760 | **selected** |
| BLOCKS=3 | 0.235714 | 0.360485 | 1.000342 |  |

Selected: ETTh2 BLOCKS=1; ETTm2 BLOCKS=2
