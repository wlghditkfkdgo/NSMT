# ETT tuning of the LIF SpkMixer, stage patch (PATCH); validation only

Rule: summarize_ettune.py docstring (score = mean over H96 / H720 of the seed-mean best validation L1 relative to the current value; lowest wins; ties keep the current value). No test values are shown at this stage.

## ETTh2 (before this stage: default (Table 13))

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| PATCH=4 | 0.314615 | 0.536550 | 1.003718 |  |
| PATCH=8 | 0.312643 | 0.535944 | 1.000000 | (current) |
| PATCH=16 | 0.310755 | 0.536659 | 0.997648 | **selected** |

## ETTm2 (before this stage: default (Table 13))

| value | validation L1 H96 | validation L1 H720 | score | |
|---|--:|--:|--:|---|
| PATCH=4 | 0.237479 | 0.361399 | 1.003025 |  |
| PATCH=8 | 0.236459 | 0.360773 | 1.000000 | (current) |
| PATCH=16 | 0.236239 | 0.360754 | 0.999507 | **selected** |

Selected: ETTh2 PATCH=16; ETTm2 PATCH=16
