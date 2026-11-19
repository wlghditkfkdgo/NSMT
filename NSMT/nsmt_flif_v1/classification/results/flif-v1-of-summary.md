# Overfitting test (Spikformer(DE) and the SpikeDE msn base; lr, head dropout, weight decay chosen on validation) — summary

See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. New runs at sha 9c1262613.

## Selections (seed 42; validation decides; test r is for reference only)

Stage S (aux form of the base, 4 cells x 3 seeds, validation relative to aux off): query pop LIF -> **auxf** (off +0.00 %, aux -0.07 %, auxf -0.10 %); query pop f-LIF -> **auxf** (off +0.00 %, aux +0.05 %, auxf -0.02 %)

| stage | model | validation s by value (%) | selected setting | edge of the range |
|---|---|---|---|---|
| A1 | Spikformer(DE) | ref 0.0005: +0.00, 0.0001: -0.04, 5e-05: +0.06 | lr 0.0001, dropout 0, wd 0.06 | no |
| A1 | query pop LIF base (SpikeDE) | ref 0.0005: +0.00, 0.0001: +0.08, 5e-05: +0.01 | lr 0.0005, dropout 0, wd 0.06 | no |
| A1 | query pop f-LIF base (SpikeDE) | ref 0.0005: +0.00, 0.0001: +0.01, 5e-05: +0.01 | lr 0.0005, dropout 0, wd 0.06 | no |
| A2 | Spikformer(DE) | ref 0: +0.00, 0.1: -0.12, 0.3: +0.01 | lr 0.0001, dropout 0.1, wd 0.06 | no |
| A2 | query pop LIF base (SpikeDE) | ref 0: +0.00, 0.1: -0.12, 0.3: -0.15 | lr 0.0005, dropout 0.3, wd 0.06 | yes |
| A2 | query pop f-LIF base (SpikeDE) | ref 0: +0.00, 0.1: -0.09, 0.3: -0.16 | lr 0.0005, dropout 0.3, wd 0.06 | yes |
| A3 | Spikformer(DE) | ref 0.06: +0.00, 0.2: +0.14, 0.5: +0.16 | lr 0.0001, dropout 0.1, wd 0.06 | no |
| A3 | query pop LIF base (SpikeDE) | ref 0.06: +0.00, 0.2: -0.04, 0.5: -0.03 | lr 0.0005, dropout 0.3, wd 0.2 | no |
| A3 | query pop f-LIF base (SpikeDE) | ref 0.06: +0.00, 0.2: -0.07, 0.5: -0.06 | lr 0.0005, dropout 0.3, wd 0.2 | no |

## Results (test MSE, mean ± sd of seeds)

| model | arm | setting | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|---|---|
| Spikformer(DE) | current | lr 0.0005, dropout 0, wd 0.06 | 0.2921 ± 0.0038 | 0.4229 ± 0.0062 | 0.1779 ± 0.0004 | 0.4048 ± 0.0023 |
| Spikformer(DE) | final | lr 0.0001, dropout 0.1, wd 0.06 | 0.2868 ± 0.0005 | 0.4222 ± 0.0015 | 0.1761 ± 0.0003 | 0.4031 ± 0.0008 |
| query pop LIF base (SpikeDE) | current | lr 0.0005, dropout 0, wd 0.06 | 0.2916 ± 0.0016 | 0.4202 ± 0.0053 | 0.1762 ± 0.0007 | 0.4023 ± 0.0009 |
| query pop LIF base (SpikeDE) | final | lr 0.0005, dropout 0.3, wd 0.2 | 0.2906 ± 0.0042 | 0.4208 ± 0.0061 | 0.1748 ± 0.0005 | 0.4034 ± 0.0004 |
| query pop f-LIF base (SpikeDE) | current | lr 0.0005, dropout 0, wd 0.06 | 0.2927 ± 0.0044 | 0.4209 ± 0.0039 | 0.1767 ± 0.0007 | 0.4040 ± 0.0017 |
| query pop f-LIF base (SpikeDE) | final | lr 0.0005, dropout 0.3, wd 0.2 | 0.2892 ± 0.0057 | 0.4204 ± 0.0061 | 0.1750 ± 0.0005 | 0.4018 ± 0.0022 |

## O1 — is the overfitting resolved? (12 runs per arm; definitions of loss_trends.py)

| model | arm | overfit flags | median best epoch | best at epoch 0 | median validation rise after best % | median epochs | resolved |
|---|---|---|---|---|---|---|---|
| Spikformer(DE) | current | 8/12 | 2.5 | 2 | +1.81 | 14 | no |
| Spikformer(DE) | final | 0/12 | 6.0 | 0 | +0.22 | 17 | yes |
| query pop LIF base (SpikeDE) | current | 8/12 | 1.5 | 3 | +1.41 | 12 | no |
| query pop LIF base (SpikeDE) | final | 5/12 | 1.5 | 1 | +0.91 | 12 | no |
| query pop f-LIF base (SpikeDE) | current | 7/12 | 1.5 | 4 | +1.46 | 12 | no |
| query pop f-LIF base (SpikeDE) | final | 3/12 | 2.0 | 3 | +0.76 | 13 | no |

## O2 / O4 (r = A / B − 1 of test MSE, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|---|
| O2 query pop LIF: base (final) − Spikformer(DE) (final) | +1.33 (worse) | -0.33 (no clear difference) | -0.72 (no clear difference) | +0.09 (no clear difference) | +0.10 | 7/12 | no clear difference | 1.26 | — |
| (reference) query pop LIF: base − Spikformer(DE), both current | -0.16 (no clear difference) | -0.62 (no clear difference) | -0.95 (no clear difference) | -0.61 (no clear difference) | -0.59 | 10/12 | no clear difference | 1.30 | — |
| O2 query pop f-LIF: base (final) − Spikformer(DE) (final) | +0.83 (no clear difference) | -0.44 (no clear difference) | -0.62 (no clear difference) | -0.30 (no clear difference) | -0.13 | 8/12 | no clear difference | 1.52 | — |
| (reference) query pop f-LIF: base − Spikformer(DE), both current | +0.22 (no clear difference) | -0.46 (no clear difference) | -0.71 (no clear difference) | -0.20 (no clear difference) | -0.29 | 8/12 | no clear difference | 1.56 | — |
| O4 Spikformer(DE): final − current | -1.81 (better) | -0.14 (no clear difference) | -1.05 (better) | -0.43 (no clear difference) | -0.86 | 10/12 | no clear difference | 1.01 | — |
| O4 query pop LIF base (SpikeDE): final − current | -0.35 (no clear difference) | +0.15 (no clear difference) | -0.81 (no clear difference) | +0.28 (no clear difference) | -0.18 | 6/12 | no clear difference | 0.98 | — |
| O4 query pop f-LIF base (SpikeDE): final − current | -1.22 (better) | -0.13 (no clear difference) | -0.96 (no clear difference) | -0.53 (no clear difference) | -0.71 | 10/12 | no clear difference | 0.99 | — |

## O3 — gap change (g = r_O2 − r_current per cell and seed, %p; negative = the base gains after the fixes)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean g | pairs negative | verdict | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|
| O3 query pop LIF | +1.49 | +0.30 | +0.24 | +0.71 | +0.68 | 3/12 | no clear change | — |
| O3 query pop f-LIF | +0.61 | +0.02 | +0.09 | -0.09 | +0.16 | 6/12 | no clear change | — |

## MIT-BIH (not tuned; the base with the stage-S aux form − Spikformer(DE); d = macro-F1 difference, %p)

| neuron | d per seed | mean | verdict | energy A/B |
|---|---|---|---|---|
| query pop LIF (auxf) | -0.19 / -0.27 / +0.07 | -0.13 | no clear difference | 1.22 |
| query pop f-LIF (auxf) | +0.28 / -0.54 / -0.32 | -0.19 | no clear difference | 1.33 |

## conclusions (rule of the pre-registration)

| comparison | Spikformer(DE) | pop LIF | pop f-LIF | conclusion |
|---|---|---|---|---|
| O1 overfitting resolved (final) | yes | no | no | (per model) |
| O2 base − Spikformer(DE), final | — | no clear difference | no clear difference | no clear difference |
| O3 gap change | — | no clear change | no clear change | no clear change |
| O4 final − current | no clear difference | no clear difference | no clear difference | (per model) |
| overall (ETT O2 + MIT-BIH) | — | not supported | not supported | — |

## report only

| model | arm | energy µJ (ETT mean) | / Spikformer(DE) same arm | rho_hippo | rho_neo | run time s (min–max) |
|---|---|---|---|---|---|---|
| Spikformer(DE) | current | 10.55 | 1.00 | 0.0608 | — | 126–1043 |
| Spikformer(DE) | final | 10.69 | 1.00 | 0.0624 | — | 145–1226 |
| query pop LIF base (SpikeDE) | current | 13.69 | 1.30 | 0.0615 | 0.1378 | 186–1054 |
| query pop LIF base (SpikeDE) | final | 13.42 | 1.26 | 0.0523 | 0.1366 | 193–1605 |
| query pop f-LIF base (SpikeDE) | current | 16.48 | 1.56 | 0.0619 | 0.1456 | 182–1036 |
| query pop f-LIF base (SpikeDE) | final | 16.22 | 1.52 | 0.0537 | 0.1443 | 193–1352 |
