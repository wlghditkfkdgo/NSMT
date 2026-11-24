# ETT: SNN baselines of the SpikF paper only (no Spikformer row); report, no training

Rules and sources: module docstring of spikf_rows_only.py. Margin = ref 4-cell mean − ours (≥ +0.0010 needed for MSE and MAE).

## first-chain final (patch 16): 4-cell MSE / MAE 0.3205 / 0.3535; energy syops µJ per cell 5.92, 21.31, 6.58, 21.94 (mean 13.94)

| reference | role | ref MSE / MAE | margin MSE / MAE | passes 0.001 rule | energy ours / ref (syops; mean of cell ratios; cells) |
|---|---|---|---|---|---|
| SpikF | SNN (SpikF paper) | 0.3215 / 0.3550 | +0.0010 / +0.0015 | yes | 0.0141 (0.007, 0.020, 0.008, 0.021) |
| SpikeRNN | SNN (SpikF paper) | 0.3270 / 0.3550 | +0.0065 / +0.0015 | yes | not available |
| iSpikformer | SNN (SpikF paper) | 0.3325 / 0.3605 | +0.0120 / +0.0070 | yes | not available |
| Informer | DNN (NSMT Table 13) | 2.7865 / 1.2352 | +2.4660 / +0.8818 | yes | not available |
| FEDformer | DNN (NSMT Table 13) | 0.3613 / 0.3932 | +0.0408 / +0.0398 | yes | not available |
| PatchTST | DNN (NSMT Table 13) | 0.3275 / 0.3632 | +0.0070 / +0.0098 | yes | not available |
| ETSformer | DNN (NSMT Table 13) | 0.3608 / 0.3953 | +0.0403 / +0.0418 | yes | not available |
| Autoformer | DNN (NSMT Table 13) | 0.3873 / 0.4175 | +0.0668 / +0.0640 | yes | not available |
| NSMT | report only (not in the SpikF paper) | 0.3177 / 0.3518 | -0.0027 / -0.0017 | no | 0.3231 (0.177, 0.475, 0.171, 0.470) |

Without criterion (2): passes every SpikF-paper SNN row and every NSMT DNN row on accuracy: **yes**. Energy vs SpikF (syops): 0.0141.
Removed criterion (2), for reference: SF-de patch 8 margin +0.0034 / +0.0022, energy ours / SF-de (syops) 0.5616 (bound was 1.05).

## extension final (patch 24): 4-cell MSE / MAE 0.3193 / 0.3526; energy syops µJ per cell 4.50, 14.76, 32.44, 73.32 (mean 31.26)

| reference | role | ref MSE / MAE | margin MSE / MAE | passes 0.001 rule | energy ours / ref (syops; mean of cell ratios; cells) |
|---|---|---|---|---|---|
| SpikF | SNN (SpikF paper) | 0.3215 / 0.3550 | +0.0022 / +0.0024 | yes | 0.0322 (0.005, 0.014, 0.040, 0.070) |
| SpikeRNN | SNN (SpikF paper) | 0.3270 / 0.3550 | +0.0077 / +0.0024 | yes | not available |
| iSpikformer | SNN (SpikF paper) | 0.3325 / 0.3605 | +0.0132 / +0.0079 | yes | not available |
| Informer | DNN (NSMT Table 13) | 2.7865 / 1.2352 | +2.4672 / +0.8827 | yes | not available |
| FEDformer | DNN (NSMT Table 13) | 0.3613 / 0.3932 | +0.0419 / +0.0407 | yes | not available |
| PatchTST | DNN (NSMT Table 13) | 0.3275 / 0.3632 | +0.0082 / +0.0107 | yes | not available |
| ETSformer | DNN (NSMT Table 13) | 0.3608 / 0.3953 | +0.0414 / +0.0427 | yes | not available |
| Autoformer | DNN (NSMT Table 13) | 0.3873 / 0.4175 | +0.0679 / +0.0649 | yes | not available |
| NSMT | report only (not in the SpikF paper) | 0.3177 / 0.3518 | -0.0016 / -0.0008 | no | 0.7191 (0.135, 0.329, 0.844, 1.569) |

Without criterion (2): passes every SpikF-paper SNN row and every NSMT DNN row on accuracy: **yes**. Energy vs SpikF (syops): 0.0322.
Removed criterion (2), for reference: SF-de patch 8 margin +0.0045 / +0.0031, energy ours / SF-de (syops) 1.3971 (bound was 1.05).

