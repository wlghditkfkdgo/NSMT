# ETT pass criteria with the NSMT-paper DNN baselines only (user decision 11-23; no training)

Rules and sources: module docstring of pass_ett_nsmt.py. Criterion (3) now = the 5 DNN rows of NSMT Table 13 (TimeMixer and the other rows not in the NSMT paper removed); (1) and (2) unchanged.

## Transcription check

- NSMT Table 13 transcription: 48 numbers checked, 0 mismatches
- NSMT column of Table 13 = pass_ett.NSMT (Table 14) on the 4 cells: True
- PatchTST of NSMT Table 13 = pass_ett "PatchTST (re-run)" (iTransformer Table 10), report: True

## NSMT-paper DNN rows (look-back 96; MSE / MAE per cell: ETTh2 H96, ETTh2 H720, ETTm2 H96, ETTm2 H720)

| row | source | MSE | MAE | 4-cell mean MSE / MAE |
|---|---|---|---|---|
| Informer | NSMT Table 13 | 3.755, 3.647, 0.365, 3.379 | 1.525, 1.625, 0.453, 1.338 | 2.7865 / 1.2352 |
| FEDformer | NSMT Table 13 | 0.358, 0.463, 0.203, 0.421 | 0.397, 0.474, 0.287, 0.415 | 0.3613 / 0.3932 |
| PatchTST | NSMT Table 13 | 0.302, 0.431, 0.175, 0.402 | 0.348, 0.446, 0.259, 0.400 | 0.3275 / 0.3632 |
| ETSformer | NSMT Table 13 | 0.340, 0.500, 0.189, 0.414 | 0.391, 0.497, 0.280, 0.413 | 0.3608 / 0.3953 |
| Autoformer | NSMT Table 13 | 0.346, 0.515, 0.255, 0.433 | 0.388, 0.511, 0.339, 0.432 | 0.3873 / 0.4175 |

Needed for (1)-(3) accuracy: 4-cell mean MSE ≤ 0.3205 and MAE ≤ 0.3540 (the binding row is now SpikF for both); plus energy A/B ≤ 1.05.

## All candidates (old = 9-row DNN set incl. TimeMixer; new = NSMT 5 rows)

| candidate | kind | configuration (non-default knobs) | MSE / MAE | energy A/B | (1) | (2) | (3) old | (3) new | overall old | overall new |
|---|---|---|---|--:|---|---|---|---|---|---|
| untuned LIF SpkMixer (11-01; reference) | reference | ETTh2 default (Table 13); ETTm2 default (Table 13) | 0.3245 / 0.3567 | 0.9825 | fail | fail | fail | pass | NOT PASSED | NOT PASSED |
| first chain final (11-20 04:02; pre-registered) | pre-registered | ETTh2 PATCH=16, WD=0.1; ETTm2 PATCH=16, MLPR=2, WD=0.1 | 0.3205 / 0.3535 | 0.5412 | pass | pass | fail | pass | NOT PASSED | PASS |
| extension final (11-20 17:07; pre-registered) | pre-registered | ETTh2 PATCH=24, WD=0.005; ETTm2 PATCH=24, EMB=256, BLOCKS=2, WD=0.02 | 0.3193 / 0.3526 | 1.3338 | pass | fail | fail | pass | NOT PASSED | NOT PASSED |
| first chain after stage patch (post hoc) | post hoc | ETTh2 PATCH=16; ETTm2 PATCH=16 | 0.3198 / 0.3531 | 0.5268 | pass | pass | fail | pass | NOT PASSED | PASS |
| first chain after stage dmodel (post hoc) | post hoc | ETTh2 PATCH=16; ETTm2 PATCH=16 | 0.3198 / 0.3531 | 0.5268 | pass | pass | fail | pass | NOT PASSED | PASS |
| first chain after stage mlp (post hoc) | post hoc | ETTh2 PATCH=16; ETTm2 PATCH=16, MLPR=2 | 0.3199 / 0.3531 | 0.5419 | pass | pass | fail | pass | NOT PASSED | PASS |
| first chain after stage blocks (post hoc) | post hoc | ETTh2 PATCH=16; ETTm2 PATCH=16, MLPR=2 | 0.3199 / 0.3531 | 0.5419 | pass | pass | fail | pass | NOT PASSED | PASS |
| extension after stage patch (post hoc) | post hoc | ETTh2 PATCH=24; ETTm2 PATCH=24 | 0.3196 / 0.3527 | 0.3753 | pass | pass | fail | pass | NOT PASSED | PASS |
| extension after stage dmodel (post hoc) | post hoc | ETTh2 PATCH=24; ETTm2 PATCH=24, EMB=256 | 0.3193 / 0.3526 | 1.0354 | pass | pass | fail | pass | NOT PASSED | PASS |
| extension after stage mlp (post hoc) | post hoc | ETTh2 PATCH=24; ETTm2 PATCH=24, EMB=256 | 0.3193 / 0.3526 | 1.0354 | pass | pass | fail | pass | NOT PASSED | PASS |
| extension after stage blocks (post hoc) | post hoc | ETTh2 PATCH=24; ETTm2 PATCH=24, EMB=256, BLOCKS=2 | 0.3194 / 0.3527 | 1.3277 | pass | fail | fail | pass | NOT PASSED | NOT PASSED |

## Detail with the NSMT set (pre-registered finals and the reference)

#### untuned LIF SpkMixer (11-01; reference)

4-cell mean MSE / MAE: **0.3245 / 0.3567**; energy A/B vs SF-de (syops) 0.9825

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | -0.0030 / -0.0017 | no |
| SF-de (ours) | 0.3238 / 0.3557 | -0.0007 / -0.0010 | no |
| energy A/B vs SF-de (syops; bound 1.05) | 0.9825 | | yes |
| Informer | 2.7865 / 1.2352 | +2.4620 / +0.8786 | yes |
| FEDformer | 0.3613 / 0.3932 | +0.0368 / +0.0366 | yes |
| PatchTST | 0.3275 / 0.3632 | +0.0030 / +0.0066 | yes |
| ETSformer | 0.3608 / 0.3953 | +0.0363 / +0.0386 | yes |
| Autoformer | 0.3873 / 0.4175 | +0.0628 / +0.0608 | yes |

- (1) SNN: **fail**; (2) Spikformer: **fail**; (3) DNN (5 rows): **pass** (5/5); overall: **NOT PASSED**

#### first chain final (11-20 04:02; pre-registered)

4-cell mean MSE / MAE: **0.3205 / 0.3535**; energy A/B vs SF-de (syops) 0.5412

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | +0.0010 / +0.0015 | yes |
| SF-de (ours) | 0.3238 / 0.3557 | +0.0034 / +0.0022 | yes |
| energy A/B vs SF-de (syops; bound 1.05) | 0.5412 | | yes |
| Informer | 2.7865 / 1.2352 | +2.4660 / +0.8818 | yes |
| FEDformer | 0.3613 / 0.3932 | +0.0408 / +0.0398 | yes |
| PatchTST | 0.3275 / 0.3632 | +0.0070 / +0.0098 | yes |
| ETSformer | 0.3608 / 0.3953 | +0.0403 / +0.0418 | yes |
| Autoformer | 0.3873 / 0.4175 | +0.0668 / +0.0640 | yes |

- (1) SNN: **pass**; (2) Spikformer: **pass**; (3) DNN (5 rows): **pass** (5/5); overall: **PASS**

#### extension final (11-20 17:07; pre-registered)

4-cell mean MSE / MAE: **0.3193 / 0.3526**; energy A/B vs SF-de (syops) 1.3338

| reference | ref 4-cell mean MSE / MAE | ref − ours MSE / MAE (≥ +0.0010 needed) | passes |
|---|---|---|---|
| SpikF | 0.3215 / 0.3550 | +0.0022 / +0.0024 | yes |
| SF-de (ours) | 0.3238 / 0.3557 | +0.0045 / +0.0031 | yes |
| energy A/B vs SF-de (syops; bound 1.05) | 1.3338 | | no |
| Informer | 2.7865 / 1.2352 | +2.4672 / +0.8827 | yes |
| FEDformer | 0.3613 / 0.3932 | +0.0419 / +0.0407 | yes |
| PatchTST | 0.3275 / 0.3632 | +0.0082 / +0.0107 | yes |
| ETSformer | 0.3608 / 0.3953 | +0.0414 / +0.0427 | yes |
| Autoformer | 0.3873 / 0.4175 | +0.0679 / +0.0649 | yes |

- (1) SNN: **pass**; (2) Spikformer: **fail**; (3) DNN (5 rows): **pass** (5/5); overall: **NOT PASSED**

Report (not a criterion): NSMT itself (Table 13 / 14) 4-cell mean MSE / MAE 0.3177 / 0.3518.
