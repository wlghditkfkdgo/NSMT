# ETT tuning of the LIF SpkMixer, stage wd (WD); validation only

Rule: summarize_ettune.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; lowest wins; ties keep the current value). No test values are shown at this stage.

## ETTh2 (before this stage: PATCH=16)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| WD=0 | 0.311249 | 0.536353 | 1.000508 |  |
| WD=0.01 | 0.311020 | 0.536307 | 1.000097 |  |
| WD=0.06 | 0.310755 | 0.536659 | 1.000000 | (current) |
| WD=0.1 | 0.310828 | 0.536321 | 0.999801 | **selected** |

## ETTm2 (before this stage: PATCH=16, MLPR=2)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| WD=0 | 0.236077 | 0.360766 | 0.999931 |  |
| WD=0.01 | 0.236132 | 0.360821 | 1.000125 |  |
| WD=0.06 | 0.236070 | 0.360826 | 1.000000 | (current) |
| WD=0.1 | 0.236039 | 0.360781 | 0.999871 | **selected** |

Selected: ETTh2 WD=0.1; ETTm2 WD=0.1
