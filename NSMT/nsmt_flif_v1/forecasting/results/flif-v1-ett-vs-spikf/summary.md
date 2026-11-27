# ETT (ETTh2 / ETTm2 x H96 / H720, input 96): current base vs SpikF and the SNN rows of SpikF Table 12

References from `origin/exp/f-lif-pop-v3` (commit 4d7ee850b); SpikF reruns: code commit ff425911c, 96 runs. Ours: flif-v1-ett-r1, sha 546d16ba5. Values: mean of seeds; 4-cell mean in the last columns. Energy = syops count (µJ).

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | 4-cell MSE | 4-cell MAE | energy µJ |
|---|---|---|---|---|---|---|---|
| SpikF (paper) | 0.2900 / 0.3360 | 0.4160 / 0.4360 | 0.1750 / 0.2510 | 0.4050 / 0.3970 | 0.3215 | 0.3550 | — |
| SpikeRNN (paper) | 0.2950 / 0.3350 | 0.4280 / 0.4380 | 0.1760 / 0.2510 | 0.4090 / 0.3960 | 0.3270 | 0.3550 | — |
| iSpikformer (paper) | 0.3080 / 0.3490 | 0.4380 / 0.4390 | 0.1780 / 0.2560 | 0.4060 / 0.3980 | 0.3325 | 0.3605 | — |
| SpikF rerun, released (3 seeds) | 0.2905 / 0.3355 | 0.4183 / 0.4358 | 0.1749 / 0.2524 | 0.4053 / 0.3965 | 0.3222 | 0.3551 | 934.59 |
| SpikF-LIF rerun (3 seeds) | 0.2861 / 0.3311 | 0.4177 / 0.4360 | 0.1728 / 0.2507 | 0.4030 / 0.3956 | 0.3199 | 0.3533 | 785.46 |
| ours, query pop LIF | 0.2902 / 0.3347 | 0.4196 / 0.4369 | 0.1756 / 0.2530 | 0.4034 / 0.3955 | 0.3222 | 0.3550 | 26.38 |
| ours, query pop f-LIF | 0.2897 / 0.3348 | 0.4207 / 0.4374 | 0.1765 / 0.2535 | 0.4044 / 0.3957 | 0.3228 | 0.3553 | 26.37 |
| Spikformer(DE) | 0.2903 / 0.3355 | 0.4249 / 0.4403 | 0.1767 / 0.2534 | 0.4048 / 0.3963 | 0.3242 | 0.3564 | 25.82 |

## Adoption bar (reference − ours ≥ 0.001 for both 4-cell MSE and MAE; energy ≤ 1.5x)

| ours | reference | margin MSE | margin MAE | energy × | cells passing both | adoption |
|---|---|---|---|---|---|---|
| ours, query pop LIF | SpikF (paper) | -0.0007 | -0.0000 | — | 1/4 | **not passed** |
| ours, query pop LIF | SpikeRNN (paper) | +0.0048 | -0.0000 | — | 1/4 | **not passed** |
| ours, query pop LIF | iSpikformer (paper) | +0.0103 | +0.0055 | — | 4/4 | **pass** |
| ours, query pop LIF | SpikF rerun, released (3 seeds) | +0.0000 | +0.0000 | 0.028 | 1/4 | **not passed** |
| ours, query pop LIF | SpikF-LIF rerun (3 seeds) | -0.0023 | -0.0017 | 0.034 | 0/4 | **not passed** |
| ours, query pop f-LIF | SpikF (paper) | -0.0013 | -0.0003 | — | 0/4 | **not passed** |
| ours, query pop f-LIF | SpikeRNN (paper) | +0.0042 | -0.0003 | — | 0/4 | **not passed** |
| ours, query pop f-LIF | iSpikformer (paper) | +0.0097 | +0.0052 | — | 4/4 | **pass** |
| ours, query pop f-LIF | SpikF rerun, released (3 seeds) | -0.0006 | -0.0003 | 0.028 | 0/4 | **not passed** |
| ours, query pop f-LIF | SpikF-LIF rerun (3 seeds) | -0.0029 | -0.0020 | 0.034 | 0/4 | **not passed** |
| Spikformer(DE) | SpikF (paper) | -0.0027 | -0.0014 | — | 0/4 | **not passed** |
| Spikformer(DE) | SpikeRNN (paper) | +0.0028 | -0.0014 | — | 0/4 | **not passed** |
| Spikformer(DE) | iSpikformer (paper) | +0.0083 | +0.0041 | — | 3/4 | **pass** |
| Spikformer(DE) | SpikF rerun, released (3 seeds) | -0.0019 | -0.0013 | 0.028 | 0/4 | **not passed** |
| Spikformer(DE) | SpikF-LIF rerun (3 seeds) | -0.0043 | -0.0031 | 0.033 | 0/4 | **not passed** |
