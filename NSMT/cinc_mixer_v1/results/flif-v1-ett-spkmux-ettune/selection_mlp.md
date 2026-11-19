# ETT tuning of the LIF SpkMixer, stage mlp (MLPR); validation only

Rule: summarize_ettune.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; lowest wins; ties keep the current value). No test values are shown at this stage.

## ETTh2 (before this stage: PATCH=16)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| MLPR=0.5 | 0.310755 | 0.536659 | 1.000000 | **selected** (current) |
| MLPR=1 | 0.310829 | 0.536951 | 1.000390 |  |
| MLPR=2 | 0.310812 | 0.536970 | 1.000380 |  |
| MLPR=4 | 0.311226 | 0.536772 | 1.000862 |  |

## ETTm2 (before this stage: PATCH=16)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| MLPR=0.5 | 0.236493 | 0.360906 | 1.000750 |  |
| MLPR=1 | 0.236239 | 0.360754 | 1.000000 | (current) |
| MLPR=2 | 0.236070 | 0.360826 | 0.999744 | **selected** |
| MLPR=4 | 0.236379 | 0.360848 | 1.000429 |  |

Selected: ETTh2 MLPR=0.5; ETTm2 MLPR=2
