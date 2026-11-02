# CinC tuning of the fusion MUX — stage res

Selection score = seed-mean best-validation CinC score (%).

| condition | candidate | validation per seed (42, 7, 2026) | mean | test mean (report) | selected |
|---|---|---|---:|---:|---|
| muxlin | or (default) | 72.84, 70.96, 73.58 | 72.46 | 68.52 | |
| muxlin | iand | 72.56, 71.64, 72.67 | 72.29 | 67.20 | |
| muxlin | add | 72.15, 70.23, 72.74 | 71.71 | 67.93 | |
| muxlin | **selected: or** | | | | yes |
| sdlin | or (default) | 69.89, 71.30, 74.88 | 72.02 | 66.96 | |
| sdlin | iand | 71.20, 70.35, 71.39 | 70.98 | 67.11 | |
| sdlin | add | 72.14, 71.05, 70.82 | 71.34 | 65.91 | |
| sdlin | **selected: or** | | | | yes |

**Selection written:** selected_res.json = {"spk_lif_muxlin_or": "or", "spk_lif_sdlin_or": "or"}
