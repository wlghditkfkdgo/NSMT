# ETT tuning of the LIF SpkMixer, stage wd (WD); validation only

Rule: summarize_ettune.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; lowest wins; ties keep the current value). No test values are shown at this stage.

## ETTh2 (before this stage: PATCH=24)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| WD=0 | 0.311225 | 0.536276 | 1.000492 |  |
| WD=0.005 | 0.311119 | 0.535589 | 0.999681 | **selected** |
| WD=0.01 | 0.311437 | 0.535839 | 1.000425 |  |
| WD=0.02 | 0.311490 | 0.536180 | 1.000829 |  |
| WD=0.05 | 0.311308 | 0.535836 | 1.000215 |  |
| WD=0.06 | 0.311171 | 0.535842 | 1.000000 | (current) |

## ETTm2 (before this stage: PATCH=24, EMB=256, BLOCKS=2)

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| WD=0 | 0.235548 | 0.360411 | 1.000127 |  |
| WD=0.005 | 0.235620 | 0.360329 | 1.000166 |  |
| WD=0.01 | 0.235675 | 0.360324 | 1.000276 |  |
| WD=0.02 | 0.235435 | 0.360359 | 0.999817 | **selected** |
| WD=0.05 | 0.235544 | 0.360344 | 1.000026 |  |
| WD=0.06 | 0.235577 | 0.360274 | 1.000000 | (current) |

Selected: ETTh2 WD=0.005; ETTm2 WD=0.02
