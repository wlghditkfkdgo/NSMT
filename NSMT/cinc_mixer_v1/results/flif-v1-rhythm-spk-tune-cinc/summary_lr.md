# CinC tuning of the fusion MUX — stage lr

Selection score = seed-mean best-validation CinC score (%).

| condition | candidate | validation per seed (42, 7, 2026) | mean | test mean (report) | selected |
|---|---|---|---:|---:|---|
| muxlin | 0.00025 | 68.85, 69.29, 70.91 | 69.68 | 64.64 | |
| muxlin | 0.0005 | 71.18, 71.56, 72.52 | 71.75 | 66.69 | |
| muxlin | 0.001 (default) | 72.84, 70.96, 73.58 | 72.46 | 68.52 | |
| muxlin | 0.002 | 71.94, 72.28, 71.06 | 71.76 | 68.97 | |
| muxlin | **selected: 0.001** | | | | yes |
| sdlin | 0.00025 | 68.53, 68.48, 68.58 | 68.53 | 64.83 | |
| sdlin | 0.0005 | 70.52, 70.24, 71.44 | 70.73 | 65.40 | |
| sdlin | 0.001 (default) | 69.89, 71.30, 74.88 | 72.02 | 66.96 | |
| sdlin | 0.002 | 71.80, 71.14, 70.90 | 71.28 | 67.61 | |
| sdlin | **selected: 0.001** | | | | yes |
| spk_ssa | 0.00025 | 65.95, 63.97, 65.30 | 65.07 | 61.78 | |
| spk_ssa | 0.0005 | 69.15, 65.48, 67.36 | 67.33 | 64.41 | |
| spk_ssa | 0.001 (default) | 71.12, 66.78, 68.46 | 68.79 | 65.60 | |
| spk_ssa | 0.002 | 71.64, 71.73, 70.43 | 71.27 | 66.36 | |
| spk_ssa | **selected: 0.002** | | | | yes |

**Selection written:** selected_lr.json = {"spk_lif_muxlin_or": 0.001, "spk_lif_sdlin_or": 0.001, "spk_ssa": 0.002}
