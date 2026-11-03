# Anomaly-detection tuning of the fusion MUX — stage layers

Selection score = seed-mean best validation MSE / that of the default (holdout: first 80 % trained, last 20 % validated; lower is better). No test values are computed in these runs.

| condition | dataset | candidate | val MSE per seed (42, 7, 2026) | mean | score | best epoch (mean) | selected |
|---|---|---|---|---:|---:|---:|---|
| muxlin | MSL | 1 | 0.05048, 0.05047, 0.04809 | 0.04968 | 1.0149 | 9.0 | |
| muxlin | MSL | 2 (default) | 0.05013, 0.04779, 0.04894 | 0.04895 | 1.0000 | 9.0 | |
| muxlin | MSL | 3 | 0.04962, 0.05322, 0.05189 | 0.05158 | 1.0536 | 9.0 | |
| muxlin | MSL | **selected: 2** | | | | | yes |
| muxlin | SMAP | 1 | 0.01469, 0.01408, 0.01485 | 0.01454 | 0.9843 | 9.0 | |
| muxlin | SMAP | 2 (default) | 0.01440, 0.01448, 0.01543 | 0.01477 | 1.0000 | 8.7 | |
| muxlin | SMAP | 3 | 0.01738, 0.01492, 0.01427 | 0.01552 | 1.0510 | 9.0 | |
| muxlin | SMAP | **selected: 1** | | | | | yes |
| muxlin | PSM | 1 | 0.01605, 0.01592, 0.01599 | 0.01599 | 1.0053 | 8.7 | |
| muxlin | PSM | 2 (default) | 0.01600, 0.01588, 0.01583 | 0.01590 | 1.0000 | 9.0 | |
| muxlin | PSM | 3 | 0.01595, 0.01548, 0.01593 | 0.01579 | 0.9927 | 8.7 | |
| muxlin | PSM | **selected: 3** | | | | | yes |
| sdlin | MSL | 1 | 0.05209, 0.05015, 0.05017 | 0.05080 | 1.0179 | 9.0 | |
| sdlin | MSL | 2 (default) | 0.04843, 0.04949, 0.05180 | 0.04991 | 1.0000 | 9.0 | |
| sdlin | MSL | 3 | 0.05253, 0.04929, 0.05043 | 0.05075 | 1.0168 | 9.0 | |
| sdlin | MSL | **selected: 2** | | | | | yes |
| sdlin | SMAP | 1 | 0.01624, 0.01509, 0.01586 | 0.01573 | 1.0646 | 8.3 | |
| sdlin | SMAP | 2 (default) | 0.01616, 0.01462, 0.01355 | 0.01478 | 1.0000 | 9.0 | |
| sdlin | SMAP | 3 | 0.01525, 0.01470, 0.01719 | 0.01571 | 1.0634 | 8.7 | |
| sdlin | SMAP | **selected: 2** | | | | | yes |
| sdlin | PSM | 1 | 0.01567, 0.01534, 0.01551 | 0.01551 | 0.9945 | 8.7 | |
| sdlin | PSM | 2 (default) | 0.01586, 0.01543, 0.01548 | 0.01559 | 1.0000 | 8.7 | |
| sdlin | PSM | 3 | 0.01535, 0.01553, 0.01573 | 0.01554 | 0.9966 | 8.7 | |
| sdlin | PSM | **selected: 1** | | | | | yes |

**Selection written:** selected_layers.json = {"spkmx_lif_muxlin_or": {"MSL": 2, "SMAP": 1, "PSM": 3}, "spkmx_lif_sdlin_or": {"MSL": 2, "SMAP": 2, "PSM": 1}}
