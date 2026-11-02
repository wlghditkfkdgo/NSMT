# MIT-BIH tuning of the fusion MUX — stage final

Tuned settings (res, lr, mixer layers): {"muxlin": ["add", 0.02, 2], "sdlin": ["or", 0.0025, 2], "spk_ssa": ["-", 0.005, 2]}

| question | mean %p | per seed (42, 7, 2026) | verdict | energy A/B | criterion (2) |
|---|---:|---|---|---:|---|
| TM1 muxlin(tuned) vs Spikformer-style(tuned) | +0.20 | -0.22, +0.39, +0.44 | no clear difference | 0.98 | not met |
| TM2 muxlin(tuned) vs sdlin(tuned) | +0.00 | -0.16, +0.18, -0.01 | no clear difference | 1.03 |  |
| TM3 sdlin(tuned) vs Spikformer-style(tuned) | +0.20 | -0.06, +0.21, +0.45 | no clear difference | 0.95 | not met |
| TM4 muxlin(tuned) vs muxlin(default) | +0.09 | -0.24, +0.11, +0.39 | no clear difference | 1.06 |  |

| condition | macro-F1 % | F1 N / S / V / F / Q | energy µJ |
|---|---:|---|---:|
| Spikformer-style (tuned) | 91.78 ± 0.25 | 99.20 / 84.38 / 96.12 / 80.17 / 99.03 | 9.61 |
| sdlin (tuned) | 91.98 ± 0.04 | 99.25 / 85.54 / 96.31 / 79.78 / 99.03 | 9.14 |
| muxlin (tuned) | 91.99 ± 0.10 | 99.24 / 85.94 / 96.31 / 79.49 / 98.95 | 9.44 |
| muxlin (default) | 91.90 ± 0.25 | 99.24 / 85.74 / 96.38 / 79.24 / 98.89 | 8.89 |
