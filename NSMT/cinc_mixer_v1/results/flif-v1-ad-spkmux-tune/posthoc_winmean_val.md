# Validation MSE of training-free reconstructions (post-hoc, report only)

Holdout validation windows (last 20 % of the training file, step 1); compare with val_mse_best of the holdout tuning runs (summary_lr.md, summary_layers.md, summary_res.md).

| dataset | validation windows | win-mean MSE | zero MSE |
|---|---:|---:|---:|
| MSL | 11565 | 0.50300 | 0.55121 |
| SMAP | 26938 | 0.44463 | 0.49633 |
| PSM | 26398 | 0.12653 | 1.67434 |
