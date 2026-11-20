# ETT tuning extension: post-hoc report (not part of the pre-registered selection or verdict)

## A. The configuration after each stage, judged with the user criteria (test values)

| after stage | ETTh2 | ETTm2 | 4-cell MSE / MAE | energy A/B vs SF-de (syops) | (1) SpikF | (2) SF-de + energy | (3) DNN | overall |
|---|---|---|---|--:|---|---|---|---|
| none (untuned) | default (Table 13) | default (Table 13) | 0.3245 / 0.3567 | 0.9825 | fail | fail | fail | NOT PASSED |
| patch | PATCH=24 | PATCH=24 | 0.3196 / 0.3527 | 0.3753 | pass | pass | fail | NOT PASSED |
| dmodel | PATCH=24 | PATCH=24, EMB=256 | 0.3193 / 0.3526 | 1.0354 | pass | pass | fail | NOT PASSED |
| mlp | PATCH=24 | PATCH=24, EMB=256 | 0.3193 / 0.3526 | 1.0354 | pass | pass | fail | NOT PASSED |
| blocks | PATCH=24 | PATCH=24, EMB=256, BLOCKS=2 | 0.3194 / 0.3527 | 1.3277 | pass | fail | fail | NOT PASSED |
| wd | PATCH=24, WD=0.005 | PATCH=24, EMB=256, BLOCKS=2, WD=0.02 | 0.3193 / 0.3526 | 1.3338 | pass | fail | fail | NOT PASSED |

## B. Energy per cell (syops rule, seed mean, µJ)

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean of cell ratios vs SF-de |
|---|--:|--:|--:|--:|--:|
| final (extension) | 5.58 | 15.84 | 41.06 | 81.93 | 1.3338 |
| after the patch stage | 5.56 | 15.83 | 5.80 | 16.06 | 0.3753 |
| SF-de | 13.84 | 44.64 | 14.80 | 45.44 | 1.0000 |

## C. Test oracle (chooses by TEST values; an upper bound of the searched space, not a valid selection)

### ETTh2: 29 configurations with all 6 runs; 5 lowest 2-cell mean test MSE

| configuration | 2-cell mean MSE | 2-cell mean MAE | energy A/B vs SF-de (2 cells) |
|---|--:|--:|--:|
| PATCH=24, BLOCKS=2 | 0.3492 | 0.3808 | 0.4762 |
| PATCH=24, MLPR=1 | 0.3497 | 0.3813 | 0.3894 |
| PATCH=16, EMB=32 | 0.3498 | 0.3819 | 0.2587 |
| PATCH=24, MLPR=4 | 0.3505 | 0.3818 | 0.4537 |
| PATCH=32 | 0.3507 | 0.3819 | 0.3022 |

### ETTm2: 29 configurations with all 6 runs; 5 lowest 2-cell mean test MSE

| configuration | 2-cell mean MSE | 2-cell mean MAE | energy A/B vs SF-de (2 cells) |
|---|--:|--:|--:|
| PATCH=24, EMB=256, MLPR=4 | 0.2868 | 0.3225 | 2.1202 |
| PATCH=24, EMB=256, MLPR=2 | 0.2868 | 0.3226 | 1.8337 |
| PATCH=24, EMB=128 | 0.2869 | 0.3224 | 0.7791 |
| PATCH=24, EMB=256, BLOCKS=2, WD=0.05 | 0.2870 | 0.3228 | 2.2781 |
| PATCH=24, EMB=256 | 0.2870 | 0.3227 | 1.6928 |

Lowest reachable 4-cell mean test MSE (best ETTh2 + best ETTm2 configuration, by test): **0.3180**; MAE (chosen separately): **0.3515**.
Same with each dataset restricted to configurations with energy A/B <= 1.05: MSE **0.3181**.
User DNN target (TimeMixer MSE 0.3167 - 0.001): MSE <= 0.3157 -> NOT reachable in the searched space, even by test choice.
