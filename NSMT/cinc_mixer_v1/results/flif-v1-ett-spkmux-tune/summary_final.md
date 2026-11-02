# ETT tuning of the fusion MUX — stage final

Validation-only selection per condition and dataset; test MSE for the verdicts. r = 100 (MSE_A − MSE_B) / MSE_B per seed.

Tuned settings: {"spkmx_lif_muxlin_or": {"ETTh2": {"lr": 0.0005, "layers": 1, "res": "iand"}, "ETTm2": {"lr": 0.0001, "layers": 1, "res": "iand"}}, "spkmx_lif_sdlin_or": {"ETTh2": {"lr": 0.00025, "layers": 1, "res": "iand"}, "ETTm2": {"lr": 0.0001, "layers": 1, "res": "iand"}}, "spk_ssa": {"ETTh2": {"lr": 0.0005, "layers": 1, "res": "-"}, "ETTm2": {"lr": 0.00025, "layers": 2, "res": "-"}}}

| question | R | pairs lower | verdict | energy A/B | criterion (2) | R on MAE |
|---|---:|---|---|---:|---|---:|
| T1 muxlin(tuned) vs Spikformer-style(tuned) (criterion 2) | +0.45 | 3/12 | no clear difference | 0.98 | not met | +0.45 |
| T2 muxlin(tuned) vs sdlin(tuned) | +0.57 | 5/12 | no clear difference | 1.00 |  | +0.27 |
| T3 sdlin(tuned) vs Spikformer-style(tuned) (criterion 2) | -0.12 | 5/12 | no clear difference | 0.98 | not met | +0.18 |
| T4 muxlin(tuned) vs muxlin(default) | -0.92 | 8/12 | no clear difference | 0.89 |  | -0.84 |

| comparison | per-cell mean r (seeds 42, 7, 2026) |
|---|---|
| T1 | ETTh2 H96 +1.67 (+2.37, +0.39, +2.25) w ; ETTh2 H720 -0.12 (+0.83, -0.55, -0.64) . ; ETTm2 H96 +0.47 (+0.19, +0.43, +0.77) . ; ETTm2 H720 -0.23 (-0.89, +0.07, +0.12) . |
| T2 | ETTh2 H96 +2.18 (+1.55, +0.53, +4.46) w ; ETTh2 H720 +0.30 (+0.45, -0.32, +0.77) . ; ETTm2 H96 +0.03 (-0.05, -0.13, +0.27) . ; ETTm2 H720 -0.22 (-0.25, +0.04, -0.46) . |
| T3 | ETTh2 H96 -0.48 (+0.81, -0.13, -2.11) . ; ETTh2 H720 -0.42 (+0.37, -0.23, -1.40) . ; ETTm2 H96 +0.44 (+0.24, +0.57, +0.50) . ; ETTm2 H720 -0.01 (-0.64, +0.03, +0.58) . |
| T4 | ETTh2 H96 +0.75 (+2.50, -1.29, +1.02) . ; ETTh2 H720 +0.12 (-1.40, +0.07, +1.70) . ; ETTm2 H96 -2.79 (-2.95, -2.89, -2.53) b ; ETTm2 H720 -1.78 (-1.23, -2.03, -2.07) b |

## Test MSE of the tuned settings (seed mean)

| condition | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---:|---:|---:|---:|
| Spikformer-style (tuned) | 0.2903 (best epoch 3.7) | 0.4223 (best epoch 3.3) | 0.1767 (best epoch 8.7) | 0.4044 (best epoch 1.3) |
| sdlin (tuned) | 0.2890 (best epoch 2.0) | 0.4205 (best epoch 1.7) | 0.1775 (best epoch 11.0) | 0.4044 (best epoch 3.0) |
| muxlin (tuned) | 0.2952 (best epoch 3.3) | 0.4218 (best epoch 1.0) | 0.1775 (best epoch 14.3) | 0.4034 (best epoch 2.7) |
| muxlin (default) | 0.2930 (best epoch 0.3) | 0.4214 (best epoch 0.3) | 0.1826 (best epoch 0.0) | 0.4107 (best epoch 0.0) |
