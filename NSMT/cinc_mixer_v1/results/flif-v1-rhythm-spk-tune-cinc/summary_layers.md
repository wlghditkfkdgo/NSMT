# CinC tuning of the fusion MUX — stage layers

Selection score = seed-mean best-validation CinC score (%).

| condition | candidate | validation per seed (42, 7, 2026) | mean | test mean (report) | selected |
|---|---|---|---:|---:|---|
| muxlin | 1 | 70.05, 69.70, 71.13 | 70.30 | 65.58 | |
| muxlin | 2 (default) | 72.84, 70.96, 73.58 | 72.46 | 68.52 | |
| muxlin | 3 | 70.71, 71.63, 71.97 | 71.44 | 67.10 | |
| muxlin | **selected: 2** | | | | yes |
| sdlin | 1 | 68.33, 70.30, 68.71 | 69.11 | 63.66 | |
| sdlin | 2 (default) | 69.89, 71.30, 74.88 | 72.02 | 66.96 | |
| sdlin | 3 | 70.15, 70.04, 69.94 | 70.04 | 66.66 | |
| sdlin | **selected: 2** | | | | yes |

**Selection written:** selected_layers.json = {"spk_lif_muxlin_or": 2, "spk_lif_sdlin_or": 2}
