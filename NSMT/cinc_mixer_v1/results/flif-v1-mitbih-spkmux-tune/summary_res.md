# MIT-BIH tuning of the fusion MUX — stage res

Selection score = seed-mean last-epoch validation macro-F1 (%; validation = fold 0 of 5 of the training file).

| condition | candidate | validation macro-F1 per seed (42, 7, 2026) | mean | selected |
|---|---|---|---:|---|
| muxlin | or (default) | 93.05, 92.55, 92.21 | 92.60 | |
| muxlin | iand | 92.08, 91.08, 92.00 | 91.72 | |
| muxlin | add | 92.84, 92.94, 92.08 | 92.62 | |
| muxlin | **selected: add** | | | yes |
| sdlin | or (default) | 93.39, 92.93, 92.73 | 93.02 | |
| sdlin | iand | 92.82, 92.17, 92.91 | 92.63 | |
| sdlin | add | 92.76, 92.62, 92.85 | 92.74 | |
| sdlin | **selected: or** | | | yes |

**Selection written:** selected_res.json = {"spkmx_lif_muxlin_or": "add", "spkmx_lif_sdlin_or": "or"}
