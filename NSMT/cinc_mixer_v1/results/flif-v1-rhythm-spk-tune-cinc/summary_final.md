# CinC tuning of the fusion MUX — stage final

Tuned settings (res, lr, mixer layers): {"muxlin": ["or", 0.001, 2], "sdlin": ["or", 0.001, 2], "spk_ssa": ["-", 0.002, 2]}

| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| TC1 muxlin(tuned) vs Spikformer-style(tuned) | +2.16 | +2.86, +2.54, +1.09 | better | 0.98 | met |
| TC2 muxlin(tuned) vs sdlin(tuned) | +1.56 | +3.64, +0.67, +0.37 | better | 1.00 |  |
| TC3 sdlin(tuned) vs Spikformer-style(tuned) | +0.61 | -0.78, +1.88, +0.72 | better | 0.99 | met |
| TC4 muxlin(tuned) vs muxlin(default) | +0.00 | +0.00, +0.00, +0.00 | no clear difference | 1.00 |  |

| condition | validation % | test score % | macro-F1 % | AF / Other / Noisy F1 | energy µJ |
|---|---:|---:|---:|---|---:|
| Spikformer-style (tuned) | 71.27 | 66.36 ± 1.04 | 62.38 | 61.18 / 54.94 / 50.43 | 131.5 |
| sdlin (tuned) | 72.02 | 66.96 ± 0.95 | 61.74 | 63.64 / 54.45 / 46.08 | 129.8 |
| muxlin (tuned) | 72.46 | 68.52 ± 0.81 | 64.35 | 65.16 / 56.88 / 51.85 | 129.4 |
| muxlin (default) | 72.46 | 68.52 ± 0.81 | 64.35 | 65.16 / 56.88 / 51.85 | 129.4 |
