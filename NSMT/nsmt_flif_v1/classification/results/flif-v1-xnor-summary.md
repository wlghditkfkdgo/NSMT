# Coupling attention test (XNOR + Log-PE vs the current score; msn base pop LIF / pop f-LIF and Spikformer-style; input 96) — summary

See the module docstring and docs/MODI_PROJECT_LOG.md for the pre-registered rules. New arms: suite flif-v1-ett-xnor, sha 7b2e186e6.

## Retrain noise (report only): the now arms retrained at the new sha (ETTh2 H96 seed 42; test MSE / MAE / validation L1)

| condition | logged (sha) | retrained (sha) | test MSE change |
|---|---|---|---|
| spk_ssa | 0.291610 / 0.336512 / 0.311086 (f0950c9a1) | 0.291610 / 0.336512 / 0.311086 (7b2e186e6) | +0.00 % |
| nsmt4q_pop_lif_msn | 0.287947 / 0.332963 / 0.312045 (16052238e) | 0.287947 / 0.332963 / 0.312045 (7b2e186e6) | +0.00 % |
| nsmt4q_pop_flif_msn | 0.286712 / 0.332825 / 0.312065 (16052238e) | 0.286712 / 0.332825 / 0.312065 (7b2e186e6) | +0.00 % |

## ETT — test MSE (mean ± sd of seeds 42 / 7 / 2026, best-validation weights)

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean validation L1 | seed sd / mean (max of cells) |
|---|---|---|---|---|---|---|
| Spikformer-style, now (dot) | 0.2903 ± 0.0012 | 0.4223 ± 0.0024 | 0.1775 ± 0.0007 | 0.4034 ± 0.0016 | 0.3612 | 0.56 % |
| Spikformer-style, XNOR + Log-PE | 0.2938 ± 0.0029 | 0.4197 ± 0.0069 | 0.1776 ± 0.0007 | 0.4041 ± 0.0017 | 0.3617 | 1.65 % |
| query pop LIF base, now (dot) | 0.2931 ± 0.0045 | 0.4157 ± 0.0011 | 0.1776 ± 0.0004 | 0.4036 ± 0.0005 | 0.3624 | 1.53 % |
| query pop LIF base, XNOR + Log-PE | 0.2927 ± 0.0042 | 0.4134 ± 0.0027 | 0.1771 ± 0.0015 | 0.4047 ± 0.0030 | 0.3622 | 1.42 % |
| query pop f-LIF base, now (dot) | 0.2904 ± 0.0039 | 0.4166 ± 0.0055 | 0.1782 ± 0.0011 | 0.4036 ± 0.0031 | 0.3618 | 1.34 % |
| query pop f-LIF base, XNOR + Log-PE | 0.2903 ± 0.0052 | 0.4202 ± 0.0027 | 0.1778 ± 0.0009 | 0.4050 ± 0.0023 | 0.3617 | 1.80 % |

## V1 / V2 / V3 (r = A / B − 1 of test MSE, %; energy = head recount, without the attention core operations)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|---|
| V1 query pop LIF: base XNOR − base now | -0.13 (no clear difference) | -0.54 (no clear difference) | -0.28 (no clear difference) | +0.26 (no clear difference) | -0.17 | 6/12 | no clear difference | 1.01 | — |
| V1 query pop f-LIF: base XNOR − base now | -0.05 (no clear difference) | +0.88 (no clear difference) | -0.23 (no clear difference) | +0.36 (no clear difference) | +0.24 | 4/12 | no clear difference | 1.01 | — |
| V2 Spikformer-style XNOR − now | +1.18 (worse) | -0.63 (no clear difference) | +0.06 (no clear difference) | +0.17 (no clear difference) | +0.20 | 4/12 | no clear difference | 1.02 | — |
| V3 query pop LIF: base XNOR − Spikformer-style XNOR | -0.36 (no clear difference) | -1.47 (better) | -0.27 (no clear difference) | +0.13 (no clear difference) | -0.49 | 9/12 | no clear difference | 1.24 | — |
| V3 query pop f-LIF: base XNOR − Spikformer-style XNOR | -1.16 (better) | +0.16 (no clear difference) | +0.13 (no clear difference) | +0.23 (no clear difference) | -0.16 | 5/12 | no clear difference | 1.51 | — |
| (reference) query pop LIF: base now − Spikformer-style now | +0.96 (no clear difference) | -1.57 (better) | +0.08 (no clear difference) | +0.04 (no clear difference) | -0.12 | 6/12 | no clear difference | 1.25 | — |
| (reference) query pop f-LIF: base now − Spikformer-style now | +0.04 (no clear difference) | -1.35 (better) | +0.43 (no clear difference) | +0.03 (no clear difference) | -0.21 | 6/12 | no clear difference | 1.51 | — |

## V4 — change of the gap (g = r_V3 − r_now per cell and seed, %p; negative = the base gains more under XNOR)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | mean g | pairs negative | verdict | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|
| V4 query pop LIF | -1.32 | +0.10 | -0.34 | +0.09 | -0.37 | 6/12 | no clear change | — |
| V4 query pop f-LIF | -1.20 | +1.51 | -0.30 | +0.20 | +0.05 | 6/12 | no clear change | — |

## conclusions (rule of the pre-registration)

| comparison | pop LIF | pop f-LIF | conclusion |
|---|---|---|---|
| V1 base XNOR − base now | no clear difference | no clear difference | no clear difference |
| V2 Spikformer-style XNOR − now | no clear difference | (one model) | no clear difference |
| V3 base XNOR − Spikformer-style XNOR | no clear difference | no clear difference | no clear difference |
| V4 gap change | no clear change | no clear change | no clear change |

Step-2 trigger (no neuron with V1 better and no neuron with V4 grew): **yes — step 2 starts**

## report only

### first-epoch stops, firing rates, energy, parameters, run time

| model | best validation at epoch 0 (of 12) | epochs run (min–max) | rho_neo mean | rho_hippo mean | energy µJ (head recount, mean of cells) | + attention core ops µJ (0.9 pJ / add) | energy / Spikformer-style now (recount; + core ops) | params ETTh2 H96 / ETTm2 H720 | run time s (min–max) |
|---|---|---|---|---|---|---|---|---|---|
| Spikformer-style, now (dot) | 2 | 11–24 | — | 0.0560 | 10.46 | 10.71 | 1.00; 1.00 | 169656 / 1132656 | 234–859 |
| Spikformer-style, XNOR + Log-PE | 3 | 11–18 | — | 0.0609 | 10.63 | 11.69 | 1.02; 1.09 | 169281 / 1131905 | 157–514 |
| query pop LIF base, now (dot) | 1 | 11–22 | 0.0328 | 0.0574 | 13.04 | 13.29 | 1.25; 1.24 | 175544 / 1143024 | 172–1179 |
| query pop LIF base, XNOR + Log-PE | 3 | 11–18 | 0.0348 | 0.0631 | 13.16 | 14.20 | 1.26; 1.33 | 175169 / 1142273 | 314–1987 |
| query pop f-LIF base, now (dot) | 2 | 11–20 | 0.0371 | 0.0581 | 15.86 | 16.11 | 1.51; 1.50 | 175544 / 1143024 | 177–1039 |
| query pop f-LIF base, XNOR + Log-PE | 2 | 11–37 | 0.0403 | 0.0641 | 16.05 | 17.09 | 1.53; 1.60 | 175169 / 1142273 | 303–3352 |

### test MAE (mean of seeds)

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 |
|---|---|---|---|---|
| Spikformer-style, now (dot) | 0.3350 | 0.4366 | 0.2536 | 0.3959 |
| Spikformer-style, XNOR + Log-PE | 0.3365 | 0.4359 | 0.2539 | 0.3964 |
| query pop LIF base, now (dot) | 0.3356 | 0.4330 | 0.2535 | 0.3954 |
| query pop LIF base, XNOR + Log-PE | 0.3356 | 0.4329 | 0.2533 | 0.3962 |
| query pop f-LIF base, now (dot) | 0.3347 | 0.4338 | 0.2540 | 0.3956 |
| query pop f-LIF base, XNOR + Log-PE | 0.3343 | 0.4363 | 0.2539 | 0.3969 |

### coupling diagnostics (coupling_diag.py; means of seeds): Neocortex / query varying share, position share, ΔMSE % when the Neocortex / position / content is removed at test time, y rate, attention-gate removal

| model | cell | varying share | position share | ΔMSE Neocortex 0 | ΔMSE position 0 | ΔMSE content 0 | y rate | gate removes |
|---|---|---|---|---|---|---|---|---|
| Spikformer-style, now (dot) | ETTh2 H96 | 0.559 | 0.441 | — | -0.07 | +1.18 | 0.0205 | 8.36 % |
| Spikformer-style, now (dot) | ETTh2 H720 | 0.533 | 0.467 | — | +0.34 | -1.00 | 0.0227 | 10.55 % |
| Spikformer-style, now (dot) | ETTm2 H96 | 0.167 | 0.833 | — | +3.29 | +1.48 | 0.0660 | 21.17 % |
| Spikformer-style, now (dot) | ETTm2 H720 | 0.267 | 0.733 | — | +0.22 | +0.79 | 0.0227 | 10.69 % |
| Spikformer-style, XNOR + Log-PE | ETTh2 H96 | 0.056 | 0.224 | — | -1.60 | -1.41 | 0.0467 | 9.15 % |
| Spikformer-style, XNOR + Log-PE | ETTh2 H720 | 0.040 | 0.234 | — | -0.52 | -1.14 | 0.0434 | 7.71 % |
| Spikformer-style, XNOR + Log-PE | ETTm2 H96 | 0.065 | 0.351 | — | -0.96 | +1.54 | 0.0783 | 20.13 % |
| Spikformer-style, XNOR + Log-PE | ETTm2 H720 | 0.055 | 0.356 | — | +0.24 | +0.22 | 0.0211 | 5.85 % |
| query pop LIF base, now (dot) | ETTh2 H96 | 0.673 | 0.327 | -1.55 | -0.17 | -1.55 | 0.0223 | 3.51 % |
| query pop LIF base, now (dot) | ETTh2 H720 | 0.505 | 0.495 | -0.51 | -0.04 | -0.51 | 0.0214 | 3.55 % |
| query pop LIF base, now (dot) | ETTm2 H96 | 0.185 | 0.813 | +1.18 | +1.44 | +1.16 | 0.0721 | 17.08 % |
| query pop LIF base, now (dot) | ETTm2 H720 | 0.333 | 0.667 | +0.61 | -0.02 | +0.61 | 0.0315 | 7.84 % |
| query pop LIF base, XNOR + Log-PE | ETTh2 H96 | 0.057 | 0.222 | +0.80 | -1.28 | -0.59 | 0.0524 | 9.33 % |
| query pop LIF base, XNOR + Log-PE | ETTh2 H720 | 0.036 | 0.230 | +0.75 | -0.36 | -0.16 | 0.0431 | 7.00 % |
| query pop LIF base, XNOR + Log-PE | ETTm2 H96 | 0.079 | 0.341 | +1.49 | -0.26 | +1.37 | 0.0746 | 17.25 % |
| query pop LIF base, XNOR + Log-PE | ETTm2 H720 | 0.064 | 0.350 | +0.54 | +0.42 | +0.54 | 0.0335 | 10.33 % |
| query pop f-LIF base, now (dot) | ETTh2 H96 | 0.611 | 0.389 | +1.20 | -0.18 | +1.20 | 0.0227 | 5.58 % |
| query pop f-LIF base, now (dot) | ETTh2 H720 | 0.566 | 0.434 | +1.52 | +0.00 | +1.52 | 0.0217 | 6.31 % |
| query pop f-LIF base, now (dot) | ETTm2 H96 | 0.160 | 0.836 | +0.18 | +2.08 | +0.24 | 0.0692 | 19.43 % |
| query pop f-LIF base, now (dot) | ETTm2 H720 | 0.336 | 0.664 | +0.49 | +0.04 | +0.49 | 0.0302 | 9.17 % |
| query pop f-LIF base, XNOR + Log-PE | ETTh2 H96 | 0.050 | 0.219 | +1.82 | -0.84 | +0.74 | 0.0464 | 7.60 % |
| query pop f-LIF base, XNOR + Log-PE | ETTh2 H720 | 0.041 | 0.230 | +1.16 | -0.99 | -1.28 | 0.0409 | 7.16 % |
| query pop f-LIF base, XNOR + Log-PE | ETTm2 H96 | 0.077 | 0.342 | +1.53 | -0.63 | +1.04 | 0.0829 | 18.36 % |
| query pop f-LIF base, XNOR + Log-PE | ETTm2 H720 | 0.065 | 0.354 | +0.24 | +0.45 | +0.47 | 0.0348 | 9.85 % |
