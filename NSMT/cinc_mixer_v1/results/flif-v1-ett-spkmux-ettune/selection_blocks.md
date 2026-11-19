# ETT tuning of the LIF SpkMixer, stage blocks (BLOCKS); validation only

Rule: summarize_ettune.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; lowest wins; ties keep the current value). No test values are shown at this stage.

## ETTh2 (before this stage: PATCH=16)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| BLOCKS=1 | 0.310755 | 0.536659 | 1.000000 | **selected** (current) |
| BLOCKS=2 | 0.310790 | 0.537196 | 1.000556 |  |
| BLOCKS=3 | 0.310946 | 0.537605 | 1.001187 |  |

## ETTm2 (before this stage: PATCH=16, MLPR=2)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| BLOCKS=1 | 0.236070 | 0.360826 | 1.000000 | **selected** (current) |
| BLOCKS=2 | 0.236237 | 0.360778 | 1.000287 |  |
| BLOCKS=3 | 0.236274 | 0.361016 | 1.000696 |  |

Selected: ETTh2 BLOCKS=1; ETTm2 BLOCKS=1
