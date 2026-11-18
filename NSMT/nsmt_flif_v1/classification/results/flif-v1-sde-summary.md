# SpikeDE-basis comparison (Spikformer-style and the msn base, every neuron from SpikeDE) — summary

See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. New runs at sha 11c83b385.

## Retrain checks (report only)

| run | logged (sha) | retrained (sha) | identical (all logged metrics) |
|---|---|---|---|
| MIT-BIH spk_ssa seed 42 | 0.924183 / 0.985563 / 9.489799 (26915f45b) | 0.924183 / 0.985563 / 9.489799 (11c83b385) | yes |
| ETTh2 H96 spkall_lif seed 42 | 0.288231 / 0.333338 / 0.312037 (1aeffcc9a) | 0.288231 / 0.333338 / 0.312037 (11c83b385) | yes |
| ETTh2 H96 spkall_lif seed 7 | 0.295817 / 0.338112 / 0.310449 (1aeffcc9a) | 0.295817 / 0.338112 / 0.310449 (11c83b385) | yes |
| ETTh2 H96 spkall_lif seed 2026 | 0.292238 / 0.336451 / 0.312673 (1aeffcc9a) | 0.292238 / 0.336451 / 0.312673 (11c83b385) | yes |

## Results (mean ± sd of seeds)

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | MIT-BIH macro-F1 % | MIT-BIH accuracy % |
|---|---|---|---|---|---|---|
| Spikformer-style, spikingjelly LIF (now) | 0.2903 ± 0.0012 | 0.4223 ± 0.0024 | 0.1775 ± 0.0007 | 0.4034 ± 0.0016 | 92.22 ± 0.21 | 98.55 |
| Spikformer-style, SpikeDE LIF | 0.2921 ± 0.0038 | 0.4229 ± 0.0062 | 0.1779 ± 0.0004 | 0.4048 ± 0.0023 | 91.94 ± 0.49 | 98.53 |
| query pop LIF base, spikingjelly hippocampus (now) | 0.2931 ± 0.0045 | 0.4157 ± 0.0011 | 0.1776 ± 0.0004 | 0.4036 ± 0.0005 | 92.16 ± 0.05 | 98.56 |
| query pop LIF base, SpikeDE everywhere | 0.2914 ± 0.0021 | 0.4163 ± 0.0051 | 0.1763 ± 0.0009 | 0.4029 ± 0.0019 | 91.90 ± 0.26 | 98.47 |
| query pop f-LIF base, spikingjelly hippocampus (now) | 0.2904 ± 0.0039 | 0.4166 ± 0.0055 | 0.1782 ± 0.0011 | 0.4036 ± 0.0031 | 92.27 ± 0.43 | 98.57 |
| query pop f-LIF base, SpikeDE everywhere | 0.2897 ± 0.0037 | 0.4181 ± 0.0065 | 0.1766 ± 0.0013 | 0.4034 ± 0.0020 | 92.01 ± 0.49 | 98.54 |

## ETT (r = A / B − 1 of test MSE, %; energy = head recount)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|---|
| E1 query pop LIF: base SpikeDE − Spikformer-style SpikeDE | -0.23 (no clear difference) | -1.55 (better) | -0.95 (no clear difference) | -0.46 (no clear difference) | -0.80 | 11/12 | no clear difference | 1.25 | — |
| E1 query pop f-LIF: base SpikeDE − Spikformer-style SpikeDE | -0.80 (no clear difference) | -1.12 (better) | -0.74 (no clear difference) | -0.33 (no clear difference) | -0.75 | 9/12 | no clear difference | 1.51 | — |
| (reference) query pop LIF: base − Spikformer-style, now | +0.96 (no clear difference) | -1.57 (better) | +0.08 (no clear difference) | +0.04 (no clear difference) | -0.12 | 6/12 | no clear difference | 1.25 | — |
| (reference) query pop f-LIF: base − Spikformer-style, now | +0.04 (no clear difference) | -1.35 (better) | +0.43 (no clear difference) | +0.03 (no clear difference) | -0.21 | 6/12 | no clear difference | 1.51 | — |
| (library) Spikformer-style SpikeDE − spikingjelly | +0.61 (no clear difference) | +0.13 (no clear difference) | +0.27 (no clear difference) | +0.34 (no clear difference) | +0.34 | 5/12 | no clear difference | 1.01 | — |
| (library) query pop LIF base SpikeDE − now | -0.58 (no clear difference) | +0.15 (no clear difference) | -0.76 (no clear difference) | -0.16 (no clear difference) | -0.34 | 9/12 | no clear difference | 1.01 | — |
| (library) query pop f-LIF base SpikeDE − now | -0.23 (no clear difference) | +0.36 (no clear difference) | -0.89 (no clear difference) | -0.03 (no clear difference) | -0.20 | 6/12 | no clear difference | 1.00 | — |

### E2 — change of the ETT gap (g = r_E1 − r_now per cell and seed, %p; negative = the base gains on the SpikeDE basis)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean g | pairs negative | verdict | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|
| E2 query pop LIF | -1.20 | +0.02 | -1.03 | -0.50 | -0.67 | 9/12 | no clear change | — |
| E2 query pop f-LIF | -0.84 | +0.23 | -1.17 | -0.37 | -0.54 | 8/12 | no clear change | — |

## MIT-BIH (d = A − B of test macro-F1, %p per seed 42 / 7 / 2026; energy = total_fft_uJ)

| comparison | d per seed | mean | verdict | energy A/B |
|---|---|---|---|---|
| M1 query pop LIF: base SpikeDE − Spikformer-style SpikeDE | -0.14 / -0.52 / +0.55 | -0.04 | no clear difference | 1.21 |
| M1 query pop f-LIF: base SpikeDE − Spikformer-style SpikeDE | -0.30 / -0.12 / +0.62 | +0.07 | no clear difference | 1.35 |
| (reference) query pop LIF: base − Spikformer-style, now | -0.32 / -0.05 / +0.17 | -0.07 | no clear difference | 1.19 |
| (reference) query pop f-LIF: base − Spikformer-style, now | -0.38 / -0.24 / +0.76 | +0.04 | no clear difference | 1.32 |
| (library) Spikformer-style SpikeDE − spikingjelly | -0.66 / +0.26 / -0.43 | -0.28 | no clear difference | 0.99 |
| (library) query pop LIF base SpikeDE − now | -0.49 / -0.22 / -0.05 | -0.25 | no clear difference | 1.00 |
| (library) query pop f-LIF base SpikeDE − now | -0.58 / +0.38 / -0.57 | -0.26 | no clear difference | 1.02 |

| comparison | g per seed | mean g | verdict |
|---|---|---|---|
| M2 query pop LIF (positive = the base gains on the SpikeDE basis) | +0.17 / -0.48 / +0.38 | +0.03 | no clear change |
| M2 query pop f-LIF (positive = the base gains on the SpikeDE basis) | +0.08 / +0.12 / -0.13 | +0.02 | no clear change |

## conclusions (rule of the pre-registration)

| comparison | pop LIF | pop f-LIF | conclusion over the two neurons |
|---|---|---|---|
| E1 ETT base − Spikformer-style (SpikeDE) | no clear difference | no clear difference | no clear difference |
| E2 ETT gap change | no clear change | no clear change | no clear change |
| M1 MIT-BIH base − Spikformer-style (SpikeDE) | no clear difference | no clear difference | no clear difference |
| M2 MIT-BIH gap change | no clear change | no clear change | no clear change |
| overall (base better than Spikformer-style on the SpikeDE basis) | not supported | weak (direction only) | — |

## report only

### energy (µJ, mean of cells / seeds) and ratios to Spikformer-style

| model | ETT energy | / Spikformer-style SpikeDE | / Spikformer-style now | MIT-BIH energy | / Spikformer-style SpikeDE | / Spikformer-style now |
|---|---|---|---|---|---|---|
| Spikformer-style, spikingjelly LIF (now) | 10.46 | 0.99 | 1.00 | 9.55 | 1.01 | 1.00 |
| Spikformer-style, SpikeDE LIF | 10.55 | 1.00 | 1.01 | 9.45 | 1.00 | 0.99 |
| query pop LIF base, spikingjelly hippocampus (now) | 13.04 | 1.24 | 1.25 | 11.37 | 1.20 | 1.19 |
| query pop LIF base, SpikeDE everywhere | 13.14 | 1.25 | 1.26 | 11.42 | 1.21 | 1.20 |
| query pop f-LIF base, spikingjelly hippocampus (now) | 15.86 | 1.50 | 1.51 | 12.57 | 1.33 | 1.32 |
| query pop f-LIF base, SpikeDE everywhere | 15.93 | 1.51 | 1.52 | 12.77 | 1.35 | 1.34 |

### firing rates, first-epoch stops (ETT), parameters, run time

| model | ETT rho_hippo | ETT rho_neo | MIT-BIH rho_hippo | MIT-BIH rho_neo | ETT best validation at epoch 0 (of 12) | ETT epochs (min–max) | params ETTh2 H96 / MIT-BIH | ETT run time s (min–max) |
|---|---|---|---|---|---|---|---|---|
| Spikformer-style, spikingjelly LIF (now) | 0.0560 | — | 0.1449 | — | 2 | 11–24 | 169656 / 904488 | 234–859 |
| Spikformer-style, SpikeDE LIF | 0.0608 | — | 0.1165 | — | 2 | 11–16 | 169656 / 904488 | 126–1043 |
| query pop LIF base, spikingjelly hippocampus (now) | 0.0574 | 0.0328 | 0.1506 | 0.0587 | 1 | 11–22 | 175544 / 1230888 | 172–1179 |
| query pop LIF base, SpikeDE everywhere | 0.0585 | 0.0332 | 0.1236 | 0.0718 | 3 | 11–18 | 175544 / 1230888 | 190–2103 |
| query pop f-LIF base, spikingjelly hippocampus (now) | 0.0581 | 0.0371 | 0.1476 | 0.0757 | 2 | 11–20 | 175544 / 1230888 | 177–1039 |
| query pop f-LIF base, SpikeDE everywhere | 0.0584 | 0.0323 | 0.1240 | 0.0967 | 4 | 11–18 | 175544 / 1230888 | 182–2095 |

### ETT test MAE (mean of seeds)

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| Spikformer-style, spikingjelly LIF (now) | 0.3350 | 0.4366 | 0.2536 | 0.3959 |
| Spikformer-style, SpikeDE LIF | 0.3360 | 0.4377 | 0.2542 | 0.3967 |
| query pop LIF base, spikingjelly hippocampus (now) | 0.3356 | 0.4330 | 0.2535 | 0.3954 |
| query pop LIF base, SpikeDE everywhere | 0.3347 | 0.4340 | 0.2529 | 0.3950 |
| query pop f-LIF base, spikingjelly hippocampus (now) | 0.3347 | 0.4338 | 0.2540 | 0.3956 |
| query pop f-LIF base, SpikeDE everywhere | 0.3347 | 0.4352 | 0.2535 | 0.3952 |
