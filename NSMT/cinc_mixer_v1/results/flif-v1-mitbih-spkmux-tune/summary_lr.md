# MIT-BIH tuning of the fusion MUX — stage lr

Selection score = seed-mean last-epoch validation macro-F1 (%; validation = fold 0 of 5 of the training file).

| condition | candidate | validation macro-F1 per seed (42, 7, 2026) | mean | selected |
|---|---|---|---:|---|
| muxlin | 0.0025 | 92.52, 92.37, 92.53 | 92.47 | |
| muxlin | 0.005 | 92.41, 91.93, 92.50 | 92.28 | |
| muxlin | 0.01 (default) | 92.69, 91.68, 92.57 | 92.31 | |
| muxlin | 0.02 | 93.05, 92.55, 92.21 | 92.60 | |
| muxlin | **selected: 0.02** | | | yes |
| sdlin | 0.0025 | 93.39, 92.93, 92.73 | 93.02 | |
| sdlin | 0.005 | 92.31, 93.20, 93.22 | 92.91 | |
| sdlin | 0.01 (default) | 92.68, 92.45, 92.55 | 92.56 | |
| sdlin | 0.02 | 92.81, 92.48, 92.45 | 92.58 | |
| sdlin | **selected: 0.0025** | | | yes |
| spk_ssa | 0.0025 | 92.15, 91.83, 91.81 | 91.93 | |
| spk_ssa | 0.005 | 92.50, 92.19, 93.10 | 92.60 | |
| spk_ssa | 0.01 (default) | 92.65, 92.34, 92.31 | 92.43 | |
| spk_ssa | 0.02 | 92.49, 92.08, 92.29 | 92.29 | |
| spk_ssa | **selected: 0.005** | | | yes |

**Selection written:** selected_lr.json = {"spkmx_lif_muxlin_or": 0.02, "spkmx_lif_sdlin_or": 0.0025, "spk_ssa": 0.005}
