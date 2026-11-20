# ETT tuning of the LIF SpkMixer: final (test values; selection was validation only) -- extended chain (ET_CHAIN=x: patch up to 32, wd within 0-0.06)

Rule and sources: summarize_ettune.py docstring. New runs in flif-v1-ett-spkmux-ettune2: 180. Seeds 42 / 7 / 2026, input 96.

## Selection path (validation score of the winner; 1 = kept the current value)

| stage | ETTh2 | ETTm2 |
|---|---|---|
| patch (PATCH) | 8 → **24** (0.9975) | 8 → **24** (0.9986) |
| dmodel (EMB) | 64 → **64** (1.0000) | 64 → **256** (0.9991) |
| mlp (MLPR) | 0.5 → **0.5** (1.0000) | 1 → **1** (1.0000) |
| blocks (BLOCKS) | 1 → **1** (1.0000) | 1 → **2** (0.9998) |
| wd (WD) | 0.06 → **0.005** (0.9997) | 0.06 → **0.02** (0.9998) |

Final configuration: ETTh2: PATCH=24, WD=0.005 (tag spkmx_lif_muxlin_iand_lr0.0005_L1_pt24_wd0.005); ETTm2: PATCH=24, EMB=256, BLOCKS=2, WD=0.02 (tag spkmx_lif_muxlin_iand_lr0.0001_L1_pt24_e256_blk2_wd0.02)

## Test MSE (seed mean), 4-cell means, energy

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean MSE | mean MAE | mean best validation L1 | energy µJ syops | energy µJ old rule | params ETTh2 H96 / ETTm2 H96 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **tuned LIF SpkMixer** | 0.2887 | 0.4141 | 0.1745 | 0.3999 | 0.3193 | 0.3526 | 0.36063 | 36.10 | 20.24 | 72000 / 995840 |
| LIF SpkMixer (untuned, 11-01) | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.3567 | 0.36145 | 29.30 | 10.28 | 169280 / 173440 |
| SF-de (lr only) | 0.2921 | 0.4229 | 0.1768 | 0.4036 | 0.3238 | 0.3557 | 0.36126 | 29.68 | 10.67 | 169656 / 174192 |

Report: the configuration after each stage (test values were not used for any selection):

| after stage | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean MSE | mean MAE | mean best validation L1 | energy µJ syops | energy µJ old rule | params ETTh2 H96 / ETTm2 H96 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| (none: untuned) | 0.2952 | 0.4218 | 0.1775 | 0.4034 | 0.3245 | 0.3567 | 0.36145 | 29.30 | 10.28 | 169280 / 173440 |
| patch | 0.2889 | 0.4144 | 0.1747 | 0.4002 | 0.3196 | 0.3527 | 0.36085 | 10.81 | 4.57 | 72000 / 76160 |
| dmodel | 0.2889 | 0.4144 | 0.1747 | 0.3994 | 0.3193 | 0.3526 | 0.36076 | 29.45 | 13.62 | 72000 / 599552 |
| mlp | 0.2889 | 0.4144 | 0.1747 | 0.3994 | 0.3193 | 0.3526 | 0.36076 | 29.45 | 13.62 | 72000 / 599552 |
| blocks | 0.2889 | 0.4144 | 0.1745 | 0.3998 | 0.3194 | 0.3527 | 0.36072 | 36.00 | 20.12 | 72000 / 995840 |
| wd | 0.2887 | 0.4141 | 0.1745 | 0.3999 | 0.3193 | 0.3526 | 0.36063 | 36.10 | 20.24 | 72000 / 995840 |

## Comparisons (ETT rule; test MSE)

| comparison | R % | lower pairs | verdict | energy A/B syops | energy A/B old rule | per cell r % | MAE R % |
|---|--:|--:|---|--:|--:|---|--:|
| tuned vs untuned LIF SpkMixer | -1.65 | 12/12 | better | 1.3615 | 1.9171 | -2.18, -1.82, -1.71, -0.87 | -1.15 |
| tuned LIF SpkMixer vs SF-de | -1.36 | 12/12 | better | 1.3338 | 1.8380 | -1.15, -2.06, -1.30, -0.91 | -0.85 |
| (report) untuned LIF SpkMixer vs SF-de | +0.30 | 5/12 | no clear difference | 0.9825 | 0.9637 | +1.08, -0.24, +0.42, -0.04 | +0.30 |
| (report) tuned (extended) vs tuned (first chain, 11-20 04:02) | -0.34 | 10/12 | no clear difference | 2.4100 | 3.0243 | -0.51, -0.24, -0.16, -0.46 | -0.26 |

## User pass criteria (results/ett-pass-20261119/pass_ett.py)

### tuned LIF SpkMixer

4-cell mean MSE / MAE: **0.3193 / 0.3526**; energy A/B vs SF-de (syops) 1.3338

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | +0.0022 / +0.0024 | yes |
| SF-de (ours) | 0.3238 / 0.3557 | +0.0045 / +0.0031 | yes |
| energy A/B vs SF-de (syops; bound 1.05) | 1.3338 | | no |
| iTransformer | 0.3290 / 0.3663 | +0.0097 / +0.0137 | yes |
| iTransformer, MAE loss | 0.3275 / 0.3623 | +0.0082 / +0.0097 | yes |
| RLinear | 0.3242 / 0.3603 | +0.0049 / +0.0077 | yes |
| PatchTST (re-run) | 0.3275 / 0.3632 | +0.0082 / +0.0107 | yes |
| PatchTST (other re-run) | 0.3347 / 0.3698 | +0.0154 / +0.0172 | yes |
| TimeMixer | 0.3167 / 0.3572 | -0.0026 / +0.0047 | no |
| TimesNet | 0.3493 / 0.3780 | +0.0299 / +0.0254 | yes |
| FITS | 0.3290 / 0.3653 | +0.0097 / +0.0127 | yes |
| DLinear (re-run) | 0.4778 / 0.4645 | +0.1584 / +0.1119 | yes |

- (1) SNN: **pass**; (2) Spikformer: **fail**; (3) DNN (all 9 rows): **fail** (8/9 rows); overall: **NOT PASSED**
- Needed for all three: 4-cell mean MSE ≤ 0.3157 (now 0.3193, short by 0.0036) and MAE ≤ 0.3540 (now 0.3526, short by 0.0000); plus energy A/B ≤ 1.05.
- Report (not a criterion): NSMT 0.3177 / 0.3518, ref − ours -0.0016 / -0.0008.

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

