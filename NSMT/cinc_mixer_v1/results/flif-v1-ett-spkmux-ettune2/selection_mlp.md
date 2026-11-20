# ETT tuning of the LIF SpkMixer, stage mlp (MLPR); validation only

Rule: summarize_ettune.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; lowest wins; ties keep the current value). No test values are shown at this stage.

## ETTh2 (before this stage: PATCH=24)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| MLPR=0.5 | 0.311171 | 0.535842 | 1.000000 | **selected** (current) |
| MLPR=1 | 0.311692 | 0.535937 | 1.000926 |  |
| MLPR=2 | 0.311480 | 0.535802 | 1.000459 |  |
| MLPR=4 | 0.311386 | 0.535635 | 1.000153 |  |

## ETTm2 (before this stage: PATCH=24, EMB=256)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| MLPR=0.5 | 0.235742 | 0.360454 | 1.000360 |  |
| MLPR=1 | 0.235556 | 0.360480 | 1.000000 | **selected** (current) |
| MLPR=2 | 0.235570 | 0.360484 | 1.000035 |  |
| MLPR=4 | 0.235697 | 0.360403 | 1.000192 |  |

Selected: ETTh2 MLPR=0.5; ETTm2 MLPR=1
