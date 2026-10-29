# Generalization batch, ETT forecasting `flif-v1-gen-ett` — summary (corrected head energy), user scope ETTh2 + ETTm2 x H96 / H720 (4 cells)

Best-validation weights (early stopping on the validation L1, patience 10) on the test set; MSE / MAE on the scaled data, mean over seeds (42, 7, 2026).

**Post-hoc energy correction (2026-10-28, not pre-registered):** total µJ of the Spikformer-style runs from `energy_recount.csv` (`energy_recount_ett.py`: the flat head counted as spike-driven ACs on each of the T_h binary steps instead of MACs on their average; same model function, MSE unchanged). Neo-mixer energies unchanged. The pre-registered numbers are in `summary.md`.

**Post-hoc recomputation (2026-10-29, not pre-registered):** the pre-registered rules applied to the 4 cells of the user's ETT scope (decision 12 of 2026-10-14) instead of the 16-cell grid; overall verdict needs >= 8 of the 12 seed pairs. Same runs, no new training. The 16-cell numbers are in `summary.md` / `summary_energyfix.md`.

## Test MSE (mean over seeds; n < 3 marked)

| condition | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 0.2903 | 0.4223 | 0.1775 | 0.4034 |
| Neo-mixer charges + |H_f - H_r|, spikeDE LIF | 0.2945 | 0.4209 | 0.1791 | 0.4048 |
| Neo-mixer charges + |H_f - H_r|, f-LIF | 0.2923 | 0.4202 | 0.1786 | 0.4052 |
| Neo-mixer charges + |H_f - H_r|, population LIF (psi 0.5) | 0.2958 | 0.4299 | 0.1796 | 0.4064 |
| Neo-mixer charges + |H_f - H_r|, population f-LIF (psi 1) | 0.2921 | 0.4217 | 0.1788 | 0.4053 |
| Neo-mixer OR, spikeDE LIF | 0.2913 | 0.4199 | 0.1789 | 0.4049 |
| Neo-mixer OR, f-LIF | 0.2921 | 0.4203 | 0.1785 | 0.4050 |
| Neo-mixer OR, population LIF (psi 0.5) | 0.2932 | 0.4206 | 0.1785 | 0.4062 |
| Neo-mixer OR, population f-LIF (psi 1) | 0.2912 | 0.4198 | 0.1780 | 0.4029 |

## Test MAE (mean over seeds; n < 3 marked)

| condition | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 0.3350 | 0.4366 | 0.2536 | 0.3959 |
| Neo-mixer charges + |H_f - H_r|, spikeDE LIF | 0.3386 | 0.4373 | 0.2564 | 0.3966 |
| Neo-mixer charges + |H_f - H_r|, f-LIF | 0.3370 | 0.4364 | 0.2563 | 0.3967 |
| Neo-mixer charges + |H_f - H_r|, population LIF (psi 0.5) | 0.3402 | 0.4412 | 0.2548 | 0.3975 |
| Neo-mixer charges + |H_f - H_r|, population f-LIF (psi 1) | 0.3376 | 0.4372 | 0.2543 | 0.3970 |
| Neo-mixer OR, spikeDE LIF | 0.3369 | 0.4363 | 0.2560 | 0.3963 |
| Neo-mixer OR, f-LIF | 0.3372 | 0.4367 | 0.2559 | 0.3967 |
| Neo-mixer OR, population LIF (psi 0.5) | 0.3388 | 0.4378 | 0.2542 | 0.3975 |
| Neo-mixer OR, population f-LIF (psi 1) | 0.3370 | 0.4368 | 0.2542 | 0.3961 |

## Energy and size (mean over the 4 cells and seeds)

| condition | runs | total µJ (FFT) per window | neocortex firing | epochs run | params (H96 / H720) |
|---|---|---|---|---|---|
| Spikformer-style SSA (T=4) | 12 | 10.46 | n/a | 15.4 | 169656–174192 / 1128120–1132656 |
| Neo-mixer charges + |H_f - H_r|, spikeDE LIF | 12 | 11.87 | 0.0484 ± 0.0115 | 15.8 | 173248–185856 / 1131712–1144320 |
| Neo-mixer charges + |H_f - H_r|, f-LIF | 12 | 12.35 | 0.0440 ± 0.0069 | 14.9 | 173248–185856 / 1131712–1144320 |
| Neo-mixer charges + |H_f - H_r|, population LIF (psi 0.5) | 12 | 13.82 | 0.0125 ± 0.0083 | 14.0 | 173760–186880 / 1132224–1145344 |
| Neo-mixer charges + |H_f - H_r|, population f-LIF (psi 1) | 12 | 17.19 | 0.0175 ± 0.0079 | 14.4 | 173760–186880 / 1132224–1145344 |
| Neo-mixer OR, spikeDE LIF | 12 | 2.13 | 0.0510 ± 0.0058 | 15.4 | 160832–173440 / 1119296–1131904 |
| Neo-mixer OR, f-LIF | 12 | 2.62 | 0.0461 ± 0.0023 | 17.0 | 160832–173440 / 1119296–1131904 |
| Neo-mixer OR, population LIF (psi 0.5) | 12 | 4.11 | 0.0401 ± 0.0251 | 15.3 | 161344–174464 / 1119808–1132928 |
| Neo-mixer OR, population f-LIF (psi 1) | 12 | 7.48 | 0.0337 ± 0.0166 | 17.5 | 161344–174464 / 1119808–1132928 |

## Pre-registered comparisons (test MSE; r = 100 (A − B) / B per seed, %)

| question | A | B | cells better / none / worse | R = mean of cell means | seed pairs A < B | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|---|
| G1 linmx LIF vs Spikformer (primary, criterion 2) | Neo-mixer charges + |H_f - H_r|, spikeDE LIF | Spikformer-style SSA (T=4) | 0 / 3 / 1 | +0.58 | 3/12 | no clear difference | 1.14 | not met |
| G2 linmx vs Spikformer (criterion 2) | Neo-mixer charges + |H_f - H_r|, f-LIF | Spikformer-style SSA (T=4) | 0 / 4 / 0 | +0.32 | 4/12 | no clear difference | 1.19 | not met |
| G2 linmx vs Spikformer (criterion 2) | Neo-mixer charges + |H_f - H_r|, population LIF (psi 0.5) | Spikformer-style SSA (T=4) | 0 / 1 / 3 | +1.40 | 2/12 | worse | 1.32 | not met |
| G2 linmx vs Spikformer (criterion 2) | Neo-mixer charges + |H_f - H_r|, population f-LIF (psi 1) | Spikformer-style SSA (T=4) | 0 / 4 / 0 | +0.41 | 5/12 | no clear difference | 1.64 | not met |
| G3 linmx vs OR (fusion) | Neo-mixer charges + |H_f - H_r|, spikeDE LIF | Neo-mixer OR, spikeDE LIF | 0 / 3 / 1 | +0.36 | 4/12 | no clear difference | 5.72 |  |
| G3 linmx vs OR (fusion) | Neo-mixer charges + |H_f - H_r|, f-LIF | Neo-mixer OR, f-LIF | 0 / 4 / 0 | +0.05 | 4/12 | no clear difference | 4.88 |  |
| G3 linmx vs OR (fusion) | Neo-mixer charges + |H_f - H_r|, population LIF (psi 0.5) | Neo-mixer OR, population LIF (psi 0.5) | 0 / 3 / 1 | +0.93 | 2/12 | no clear difference | 3.53 |  |
| G3 linmx vs OR (fusion) | Neo-mixer charges + |H_f - H_r|, population f-LIF (psi 1) | Neo-mixer OR, population f-LIF (psi 1) | 0 / 4 / 0 | +0.45 | 3/12 | no clear difference | 2.42 |  |
| G4 OR population LIF vs Spikformer (criterion 2) | Neo-mixer OR, population LIF (psi 0.5) | Spikformer-style SSA (T=4) | 0 / 3 / 1 | +0.47 | 4/12 | no clear difference | 0.39 | met |
| G5 (report) OR vs Spikformer (criterion 2) | Neo-mixer OR, spikeDE LIF | Spikformer-style SSA (T=4) | 0 / 4 / 0 | +0.22 | 4/12 | no clear difference | 0.20 | met |
| G5 (report) OR vs Spikformer (criterion 2) | Neo-mixer OR, f-LIF | Spikformer-style SSA (T=4) | 0 / 4 / 0 | +0.28 | 5/12 | no clear difference | 0.25 | met |
| G5 (report) OR vs Spikformer (criterion 2) | Neo-mixer OR, population f-LIF (psi 1) | Spikformer-style SSA (T=4) | 0 / 4 / 0 | -0.04 | 6/12 | no clear difference | 0.71 | not met |

## Per-cell mean r (%) and cell verdict (b = better, w = worse, . = no clear difference)

| A vs B | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| mixlinmx_lif − spk_ssa | +1.4 w | -0.3 . | +0.9 . | +0.3 . |
| mixlinmx_flif − spk_ssa | +0.7 . | -0.5 . | +0.7 . | +0.4 . |
| mixlinmx_pop_lif − spk_ssa | +1.9 w | +1.8 w | +1.2 w | +0.7 . |
| mixlinmx_pop_flif − spk_ssa | +0.6 . | -0.1 . | +0.7 . | +0.4 . |
| mixlinmx_lif − mix_lif_bidir | +1.1 w | +0.3 . | +0.1 . | -0.0 . |
| mixlinmx_flif − mix_flif_bidir | +0.1 . | -0.0 . | +0.1 . | +0.0 . |
| mixlinmx_pop_lif − mix_pop_lif_bidir | +0.9 . | +2.2 w | +0.6 . | +0.0 . |
| mixlinmx_pop_flif − mix_pop_flif_bidir | +0.3 . | +0.5 . | +0.5 . | +0.6 . |
| mix_pop_lif_bidir − spk_ssa | +1.0 w | -0.4 . | +0.6 . | +0.7 . |
| mix_lif_bidir − spk_ssa | +0.3 . | -0.6 . | +0.8 . | +0.4 . |
| mix_flif_bidir − spk_ssa | +0.6 . | -0.5 . | +0.6 . | +0.4 . |
| mix_pop_flif_bidir − spk_ssa | +0.3 . | -0.6 . | +0.3 . | -0.1 . |
