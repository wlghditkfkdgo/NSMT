# MIT-BIH tuning of the fusion MUX — stage layers

Selection score = seed-mean last-epoch validation macro-F1 (%; validation = fold 0 of 5 of the training file).

| condition | candidate | validation macro-F1 per seed (42, 7, 2026) | mean | selected |
|---|---|---|---:|---|
| muxlin | 1 | 91.80, 92.17, 92.21 | 92.06 | |
| muxlin | 2 (default) | 93.05, 92.55, 92.21 | 92.60 | |
| muxlin | 3 | 91.78, 92.38, 92.60 | 92.25 | |
| muxlin | **selected: 2** | | | yes |
| sdlin | 1 | 92.30, 92.51, 92.14 | 92.32 | |
| sdlin | 2 (default) | 93.39, 92.93, 92.73 | 93.02 | |
| sdlin | 3 | 92.66, 92.34, 92.03 | 92.34 | |
| sdlin | **selected: 2** | | | yes |

**Selection written:** selected_layers.json = {"spkmx_lif_muxlin_or": 2, "spkmx_lif_sdlin_or": 2}
