# Paper bar under the syops library rule as published

Rules and sources: module docstring of paper_bar_syops.py. Seeds 42 / 7 / 2026. Performance values are unchanged; only the energy rule differs.

AD tuned settings (11-03, validation): {"MSL": "or / lr 0.002 / M 2", "SMAP": "add / lr 0.002 / M 1", "PSM": "add / lr 0.001 / M 3"}

| task | comparison | performance (seed mean) | mean better | energy A/B current rule | paper bar current | energy A/B syops | paper bar syops | A / B µJ current | A / B µJ syops |
|---|---|---|---|--:|---|--:|---|---|---|
| ETT input 96 | LIF SpkMixer vs SF-de | R +0.305 % (MSE; < 0 better) | no | 0.9637 | not met | 0.9825 | not met | 10.28 / 10.67 | 29.30 / 29.68 |
| ETT input 336 | LIF SpkMixer vs SF-de | R -0.254 % (MSE; < 0 better) | yes | 0.9740 | met | 0.9889 | met | 34.69 / 35.61 | 102.55 / 103.39 |
| MIT-BIH | LIF SpkMixer (ADD) vs SF-de | +0.042 %p (macro-F1; > 0 better) | yes | 0.9993 | met | 4.1152 | not met | 9.44 / 9.45 | 38.87 / 9.44 |
| AD | muxlin(tuned) vs SF-de | R +1.368 % (AUC-PR; > 0 better) | yes | 1.0499 | not met | 1.1937 | not met | 10.12 / 9.72 | 12.19 / 10.62 |
| AD | muxlin(default: OR / 5e-4 / M 2) vs SF-de | R +2.648 % (AUC-PR; > 0 better) | yes | 1.0043 | not met | 1.0097 | not met | 9.75 / 9.72 | 10.69 / 10.62 |

Energy A/B is the mean over cells (ETT) / datasets (AD) of the per-cell ratio of seed means, as in the summarizers.

- AD muxlin(tuned) vs SF-de per dataset (current -> syops): MSL 1.009 -> 1.012; SMAP 0.982 -> 1.221; PSM 1.158 -> 1.348
- AD muxlin(default: OR / 5e-4 / M 2) vs SF-de per dataset (current -> syops): MSL 0.998 -> 1.000; SMAP 1.005 -> 1.015; PSM 1.010 -> 1.015
