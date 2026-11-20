# ETT tuning of the LIF SpkMixer, stage dmodel (EMB); validation only

Rule: summarize_ettune.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; lowest wins; ties keep the current value). No test values are shown at this stage.

## ETTh2 (before this stage: PATCH=24)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| EMB=32 | 0.311451 | 0.536427 | 1.000997 |  |
| EMB=64 | 0.311171 | 0.535842 | 1.000000 | **selected** (current) |
| EMB=128 | 0.310791 | 0.537029 | 1.000497 |  |
| EMB=256 | 0.311965 | 0.538293 | 1.003563 |  |

## ETTm2 (before this stage: PATCH=24)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| EMB=32 | 0.236739 | 0.360517 | 1.001674 |  |
| EMB=64 | 0.236096 | 0.360293 | 1.000000 | (current) |
| EMB=128 | 0.235719 | 0.360360 | 0.999296 |  |
| EMB=256 | 0.235556 | 0.360480 | 0.999116 | **selected** |

Selected: ETTh2 EMB=64; ETTm2 EMB=256
