# ETT tuning of the LIF SpkMixer, stage dmodel (EMB); validation only

Rule: summarize_ettune.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; lowest wins; ties keep the current value). No test values are shown at this stage.

## ETTh2 (before this stage: PATCH=16)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| EMB=32 | 0.311428 | 0.536615 | 1.001041 |  |
| EMB=64 | 0.310755 | 0.536659 | 1.000000 | **selected** (current) |
| EMB=128 | 0.311298 | 0.537417 | 1.001579 |  |
| EMB=256 | 0.313445 | 0.539123 | 1.006623 |  |

## ETTm2 (before this stage: PATCH=16)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| EMB=32 | 0.236752 | 0.360992 | 1.001417 |  |
| EMB=64 | 0.236239 | 0.360754 | 1.000000 | **selected** (current) |
| EMB=128 | 0.236383 | 0.360952 | 1.000580 |  |
| EMB=256 | 0.236279 | 0.361026 | 1.000462 |  |

Selected: ETTh2 EMB=64; ETTm2 EMB=64
