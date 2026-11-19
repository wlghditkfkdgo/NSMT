# ETT tuning of the LIF SpkMixer: final (test values; selection was validation only)

Rule and sources: summarize_ettune.py docstring. New runs in flif-v1-ett-spkmux-ettune: 156. Seeds 42 / 7 / 2026, input 96.

## Selection path (validation score of the winner; 1 = kept the current value)

| stage | ETTh2 | ETTm2 |
|---|---|---|
| patch (PATCH) | 8 → **16** (0.9976) | 8 → **16** (0.9995) |
| dmodel (EMB) | 64 → **64** (1.0000) | 64 → **64** (1.0000) |
| mlp (MLPR) | 0.5 → **0.5** (1.0000) | 1 → **2** (0.9997) |
| blocks (BLOCKS) | 1 → **1** (1.0000) | 1 → **1** (1.0000) |
| wd (WD) | 0.06 → **0.1** (0.9998) | 0.06 → **0.1** (0.9999) |

Final configuration: ETTh2: PATCH=16, WD=0.1 (tag spkmx_lif_muxlin_iand_lr0.0005_L1_pt16_wd0.1); ETTm2: PATCH=16, MLPR=2, WD=0.1 (tag spkmx_lif_muxlin_iand_lr0.0001_L1_pt16_mlp2_wd0.1)

## Test MSE (seed mean), 4-cell means, energy

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean MSE | mean MAE | mean best validation L1 | energy µJ syops | energy µJ old rule | params ETTh2 H96 / ETTm2 H96 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **tuned LIF SpkMixer** | 0.2902 | 0.4151 | 0.1748 | 0.4018 | 0.3205 | 0.3535 | 0.36099 | 15.75 | 6.34 | 96064 / 108544 |
| LIF SpkMixer (untuned, 11-01) | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.3567 | 0.36145 | 29.30 | 10.28 | 169280 / 173440 |
| SF-de (lr only) | 0.2921 | 0.4229 | 0.1768 | 0.4036 | 0.3238 | 0.3557 | 0.36126 | 29.68 | 10.67 | 169656 / 174192 |

Report: the configuration after each stage (test values were not used for any selection):

| after stage | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean MSE | mean MAE | mean best validation L1 | energy µJ syops | energy µJ old rule | params ETTh2 H96 / ETTm2 H96 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| (none: untuned) | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.3567 | 0.36145 | 29.30 | 10.28 | 169280 / 173440 |
| patch | 0.2871 | 0.4157 | 0.1749 | 0.4017 | 0.3198 | 0.3531 | 0.36110 | 15.43 | 6.03 | 96064 / 100224 |
| dmodel | 0.2871 | 0.4157 | 0.1749 | 0.4017 | 0.3198 | 0.3531 | 0.36110 | 15.43 | 6.03 | 96064 / 100224 |
| mlp | 0.2871 | 0.4157 | 0.1748 | 0.4020 | 0.3199 | 0.3531 | 0.36108 | 15.76 | 6.37 | 96064 / 108544 |
| blocks | 0.2871 | 0.4157 | 0.1748 | 0.4020 | 0.3199 | 0.3531 | 0.36108 | 15.76 | 6.37 | 96064 / 108544 |
| wd | 0.2902 | 0.4151 | 0.1748 | 0.4018 | 0.3205 | 0.3535 | 0.36099 | 15.75 | 6.34 | 96064 / 108544 |

## Comparisons (ETT rule; test MSE)

| comparison | R % | lower pairs | verdict | energy A/B syops | energy A/B old rule | per cell r % | MAE R % |
|---|--:|--:|---|--:|--:|---|--:|
| tuned vs untuned LIF SpkMixer | -1.31 | 12/12 | better | 0.5511 | 0.6158 | -1.69, -1.58, -1.56, -0.41 | -0.89 |
| tuned LIF SpkMixer vs SF-de | -1.02 | 11/12 | better | 0.5412 | 0.5933 | -0.65, -1.82, -1.14, -0.46 | -0.59 |
| (report) untuned LIF SpkMixer vs SF-de | +0.30 | 5/12 | no clear difference | 0.9825 | 0.9637 | +1.08, -0.24, +0.42, -0.04 | +0.30 |

## User pass criteria (results/ett-pass-20261119/pass_ett.py)

### tuned LIF SpkMixer

4-cell mean MSE / MAE: **0.3205 / 0.3535**; energy A/B vs SF-de (syops) 0.5412

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | +0.0010 / +0.0015 | yes |
| SF-de (ours) | 0.3238 / 0.3557 | +0.0034 / +0.0022 | yes |
| energy A/B vs SF-de (syops; bound 1.05) | 0.5412 | | yes |
| iTransformer | 0.3290 / 0.3663 | +0.0085 / +0.0128 | yes |
| iTransformer, MAE loss | 0.3275 / 0.3623 | +0.0070 / +0.0088 | yes |
| RLinear | 0.3242 / 0.3603 | +0.0038 / +0.0068 | yes |
| PatchTST (re-run) | 0.3275 / 0.3632 | +0.0070 / +0.0098 | yes |
| PatchTST (other re-run) | 0.3347 / 0.3698 | +0.0143 / +0.0163 | yes |
| TimeMixer | 0.3167 / 0.3572 | -0.0037 / +0.0038 | no |
| TimesNet | 0.3493 / 0.3780 | +0.0288 / +0.0245 | yes |
| FITS | 0.3290 / 0.3653 | +0.0085 / +0.0118 | yes |
| DLinear (re-run) | 0.4778 / 0.4645 | +0.1573 / +0.1110 | yes |

- (1) SNN: **pass**; (2) Spikformer: **pass**; (3) DNN (all 9 rows): **fail** (8/9 rows); overall: **NOT PASSED**
- Needed for all three: 4-cell mean MSE ≤ 0.3157 (now 0.3205, short by 0.0047) and MAE ≤ 0.3540 (now 0.3535, short by 0.0000); plus energy A/B ≤ 1.05.
- Report (not a criterion): NSMT 0.3177 / 0.3518, ref − ours -0.0027 / -0.0017.

### untuned LIF SpkMixer (reference)

4-cell mean MSE / MAE: **0.3245 / 0.3567**; energy A/B vs SF-de (syops) 0.9825

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | -0.0030 / -0.0017 | no |
| SF-de (ours) | 0.3238 / 0.3557 | -0.0007 / -0.0010 | no |
| energy A/B vs SF-de (syops; bound 1.05) | 0.9825 | | yes |
| iTransformer | 0.3290 / 0.3663 | +0.0045 / +0.0096 | yes |
| iTransformer, MAE loss | 0.3275 / 0.3623 | +0.0030 / +0.0056 | yes |
| RLinear | 0.3242 / 0.3603 | -0.0002 / +0.0036 | no |
| PatchTST (re-run) | 0.3275 / 0.3632 | +0.0030 / +0.0066 | yes |
| PatchTST (other re-run) | 0.3347 / 0.3698 | +0.0103 / +0.0131 | yes |
| TimeMixer | 0.3167 / 0.3572 | -0.0077 / +0.0006 | no |
| TimesNet | 0.3493 / 0.3780 | +0.0248 / +0.0213 | yes |
| FITS | 0.3290 / 0.3653 | +0.0045 / +0.0086 | yes |
| DLinear (re-run) | 0.4778 / 0.4645 | +0.1533 / +0.1078 | yes |

- (1) SNN: **fail**; (2) Spikformer: **fail**; (3) DNN (all 9 rows): **fail** (7/9 rows); overall: **NOT PASSED**
- Needed for all three: 4-cell mean MSE ≤ 0.3157 (now 0.3245, short by 0.0087) and MAE ≤ 0.3540 (now 0.3567, short by 0.0027); plus energy A/B ≤ 1.05.
- Report (not a criterion): NSMT 0.3177 / 0.3518, ref − ours -0.0067 / -0.0049.

