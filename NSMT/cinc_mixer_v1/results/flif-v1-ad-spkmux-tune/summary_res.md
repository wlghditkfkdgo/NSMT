# Anomaly-detection tuning of the fusion MUX — stage res

Selection score = seed-mean best validation MSE / that of the default (holdout: first 80 % trained, last 20 % validated; lower is better). No test values are computed in these runs.

| condition | dataset | candidate | val MSE per seed (42, 7, 2026) | mean | score | best epoch (mean) | selected |
|---|---|---|---|---:|---:|---:|---|
| muxlin | MSL | or (default) | 0.05013, 0.04779, 0.04894 | 0.04895 | 1.0000 | 9.0 | |
| muxlin | MSL | iand | 0.09889, 0.07476, 0.08753 | 0.08706 | 1.7783 | 5.7 | |
| muxlin | MSL | add | 0.05131, 0.05240, 0.05275 | 0.05215 | 1.0654 | 9.0 | |
| muxlin | MSL | **selected: or** | | | | | yes |
| muxlin | SMAP | or (default) | 0.01469, 0.01408, 0.01485 | 0.01454 | 1.0000 | 9.0 | |
| muxlin | SMAP | iand | 0.02012, 0.01837, 0.01880 | 0.01909 | 1.3135 | 9.0 | |
| muxlin | SMAP | add | 0.01188, 0.01178, 0.01210 | 0.01192 | 0.8201 | 9.0 | |
| muxlin | SMAP | **selected: add** | | | | | yes |
| muxlin | PSM | or (default) | 0.01595, 0.01548, 0.01593 | 0.01579 | 1.0000 | 8.7 | |
| muxlin | PSM | iand | 0.01499, 0.01516, 0.01512 | 0.01509 | 0.9560 | 8.0 | |
| muxlin | PSM | add | 0.01451, 0.01485, 0.01470 | 0.01468 | 0.9301 | 8.7 | |
| muxlin | PSM | **selected: add** | | | | | yes |
| sdlin | MSL | or (default) | 0.04843, 0.04949, 0.05180 | 0.04991 | 1.0000 | 9.0 | |
| sdlin | MSL | iand | 0.07868, 0.08346, 0.08195 | 0.08136 | 1.6302 | 8.7 | |
| sdlin | MSL | add | 0.05149, 0.04714, 0.05264 | 0.05043 | 1.0104 | 7.3 | |
| sdlin | MSL | **selected: or** | | | | | yes |
| sdlin | SMAP | or (default) | 0.01616, 0.01462, 0.01355 | 0.01478 | 1.0000 | 9.0 | |
| sdlin | SMAP | iand | 0.01748, 0.01747, 0.01770 | 0.01755 | 1.1877 | 9.0 | |
| sdlin | SMAP | add | 0.01201, 0.01193, 0.01242 | 0.01212 | 0.8202 | 9.0 | |
| sdlin | SMAP | **selected: add** | | | | | yes |
| sdlin | PSM | or (default) | 0.01567, 0.01534, 0.01551 | 0.01551 | 1.0000 | 8.7 | |
| sdlin | PSM | iand | 0.01528, 0.01534, 0.01524 | 0.01529 | 0.9859 | 7.3 | |
| sdlin | PSM | add | 0.01344, 0.01306, 0.01381 | 0.01344 | 0.8664 | 9.0 | |
| sdlin | PSM | **selected: add** | | | | | yes |

**Selection written:** selected_res.json = {"spkmx_lif_muxlin_or": {"MSL": "or", "SMAP": "add", "PSM": "add"}, "spkmx_lif_sdlin_or": {"MSL": "or", "SMAP": "add", "PSM": "add"}}
