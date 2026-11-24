# hyperparameter range 2 (lr x batch size x warmup) — summary

Pre-registered rules: docs/MODI_PROJECT_LOG.md 2026-11-23 22:37 KST. Search runs at sha 2ee28acde; final runs: ETT 54, MIT-BIH 15 at sha baa782c76. Seed-42 ETT final runs that repeat their search run exactly: 18/18. Search rows recovered from their run folders after a SKIP line (recover_r2_rows.py, collected=1): 5.

## selected settings (validation only, seed 42; * = range edge)

| model | ETTh2 | score % (best new) | ETTm2 | score % (best new) | MIT-BIH | validation macro-F1 % (best new / current) |
|---|---|---|---|---|---|---|
| Spikformer(DE) | lr 0.00025*, batch 32*, warmup 1 | -0.477 | lr 0.0005, batch 32*, warmup none | -0.059 | lr 0.02*, batch 256*, warmup 6 | 92.61 / 91.36 |
| base pop LIF | lr 0.0005, batch 64, warmup 1 | -0.376 | lr 0.0005, batch 32*, warmup 1 | -0.025 | lr 0.01, batch 512, warmup 0 | 92.24 / 91.95 |
| (나) pop LIF | lr 0.001*, batch 128*, warmup none | -0.128 | lr 0.0005, batch 32*, warmup 1 | -0.139 | lr 0.02*, batch 1024*, warmup 0 | 92.51 / 91.59 |
| base pop f-LIF | lr 0.00025*, batch 32*, warmup 1 | -0.243 | current kept | +0.007 | lr 0.005*, batch 256*, warmup 0 | 92.64 / 91.60 |
| (나) pop f-LIF | lr 0.001*, batch 64, warmup none | -0.062 | lr 0.00025*, batch 32*, warmup none | -0.150 | lr 0.005*, batch 256*, warmup 0 | 92.42 / 91.76 |

## results at the final settings (mean ± sd of seeds; ETT test MSE, MIT-BIH test macro-F1 %)

| model | setting | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | MIT-BIH macro-F1 % |
|---|---|---|---|---|---|---|
| Spikformer(DE) | current | 0.2903 ± 0.0032 | 0.4249 ± 0.0087 | 0.1767 ± 0.0002 | 0.4048 ± 0.0015 | 91.20 ± 0.29 |
| Spikformer(DE) | final | 0.2897 ± 0.0034 | 0.4214 ± 0.0039 | 0.1763 ± 0.0006 | 0.4044 ± 0.0013 | 91.16 ± 0.37 |
| base pop LIF | current | 0.2902 ± 0.0018 | 0.4196 ± 0.0057 | 0.1756 ± 0.0007 | 0.4034 ± 0.0007 | 91.39 ± 0.41 |
| base pop LIF | final | 0.2898 ± 0.0008 | 0.4199 ± 0.0045 | 0.1759 ± 0.0007 | 0.4031 ± 0.0003 | 91.45 ± 0.58 |
| (나) pop LIF | current | 0.2909 ± 0.0043 | 0.4235 ± 0.0066 | 0.1761 ± 0.0011 | 0.4023 ± 0.0019 | 91.46 ± 0.31 |
| (나) pop LIF | final | 0.2915 ± 0.0033 | 0.4267 ± 0.0080 | 0.1759 ± 0.0005 | 0.4009 ± 0.0005 | 90.79 ± 0.13 |
| base pop f-LIF | current | 0.2897 ± 0.0047 | 0.4207 ± 0.0074 | 0.1765 ± 0.0009 | 0.4044 ± 0.0012 | 91.57 ± 0.14 |
| base pop f-LIF | final | 0.2871 ± 0.0030 | 0.4182 ± 0.0048 | 0.1765 ± 0.0009 | 0.4044 ± 0.0012 | 91.54 ± 0.37 |
| (나) pop f-LIF | current | 0.2926 ± 0.0031 | 0.4224 ± 0.0045 | 0.1766 ± 0.0008 | 0.4030 ± 0.0011 | 91.02 ± 0.11 |
| (나) pop f-LIF | final | 0.2958 ± 0.0043 | 0.4277 ± 0.0047 | 0.1761 ± 0.0008 | 0.4029 ± 0.0014 | 91.00 ± 0.36 |

## ETT (r = A / B − 1 of test MSE, %)

| comparison | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | R % | pairs A lower | verdict | energy A/B | leave-one-cell-out |
|---|---|---|---|---|---|---|---|---|---|
| T Spikformer(DE): final − current | -0.17 (no clear difference) | -0.82 (no clear difference) | -0.23 (no clear difference) | -0.09 (no clear difference) | -0.33 | 9/12 | no clear difference | 1.00 | — |
| T base pop LIF: final − current | -0.14 (no clear difference) | +0.08 (no clear difference) | +0.14 (no clear difference) | -0.09 (no clear difference) | -0.00 | 7/12 | no clear difference | 1.00 | — |
| T (나) pop LIF: final − current | +0.23 (no clear difference) | +0.77 (no clear difference) | -0.09 (no clear difference) | -0.36 (no clear difference) | +0.14 | 8/12 | no clear difference | 0.99 | — |
| T base pop f-LIF: final − current | -0.87 (no clear difference) | -0.59 (no clear difference) | +0.00 (no clear difference) | +0.00 (no clear difference) | -0.37 | 5/12 | no clear difference | 1.00 | — |
| T (나) pop f-LIF: final − current | +1.10 (worse) | +1.24 (worse) | -0.29 (no clear difference) | -0.03 (no clear difference) | +0.51 | 5/12 | no clear difference | 0.99 | — |
| S1 pop LIF: (나) − base, final | +0.59 (no clear difference) | +1.63 (worse) | +0.05 (no clear difference) | -0.55 (no clear difference) | +0.43 | 7/12 | no clear difference | 0.63 | — |
| S1 pop f-LIF: (나) − base, final | +3.04 (worse) | +2.27 (worse) | -0.19 (no clear difference) | -0.39 (no clear difference) | +1.18 | 6/12 | no clear difference | 0.70 | — |
| S2 pop LIF: base − Spikformer(DE), final | +0.04 (no clear difference) | -0.35 (no clear difference) | -0.24 (no clear difference) | -0.34 (no clear difference) | -0.22 | 10/12 | no clear difference | 1.24 | — |
| S2 pop LIF: (나) − Spikformer(DE), final | +0.61 (no clear difference) | +1.26 (worse) | -0.19 (no clear difference) | -0.89 (no clear difference) | +0.20 | 7/12 | no clear difference | 0.79 | — |
| S2 pop f-LIF: base − Spikformer(DE), final | -0.88 (no clear difference) | -0.76 (no clear difference) | +0.11 (no clear difference) | -0.01 (no clear difference) | -0.39 | 9/12 | no clear difference | 1.50 | — |
| S2 pop f-LIF: (나) − Spikformer(DE), final | +2.10 (worse) | +1.49 (worse) | -0.09 (no clear difference) | -0.39 (no clear difference) | +0.78 | 5/12 | no clear difference | 1.05 | — |
| (reference) pop LIF: (나) − base, current | +0.22 (no clear difference) | +0.94 (no clear difference) | +0.28 (no clear difference) | -0.28 (no clear difference) | +0.29 | 7/12 | no clear difference | 0.64 | — |
| (reference) pop LIF: base − Spikformer(DE), current | -0.02 (no clear difference) | -1.25 (better) | -0.60 (no clear difference) | -0.34 (no clear difference) | -0.55 | 9/12 | no clear difference | 1.24 | — |
| (reference) pop LIF: (나) − Spikformer(DE), current | +0.21 (no clear difference) | -0.32 (no clear difference) | -0.33 (no clear difference) | -0.62 (no clear difference) | -0.26 | 9/12 | no clear difference | 0.79 | — |
| (reference) pop f-LIF: (나) − base, current | +1.02 (worse) | +0.42 (no clear difference) | +0.10 (no clear difference) | -0.36 (no clear difference) | +0.30 | 6/12 | no clear difference | 0.71 | — |
| (reference) pop f-LIF: base − Spikformer(DE), current | -0.19 (no clear difference) | -0.99 (no clear difference) | -0.12 (no clear difference) | -0.10 (no clear difference) | -0.35 | 9/12 | no clear difference | 1.50 | — |
| (reference) pop f-LIF: (나) − Spikformer(DE), current | +0.82 (no clear difference) | -0.57 (no clear difference) | -0.03 (no clear difference) | -0.46 (no clear difference) | -0.06 | 8/12 | no clear difference | 1.06 | — |

## MIT-BIH (d = A − B of test macro-F1, %p per seed 42 / 7 / 2026)

| comparison | d per seed | mean | verdict | energy A/B |
|---|---|---|---|---|
| T Spikformer(DE): final − current | -0.61 / -0.19 / +0.69 | -0.04 | no clear difference | 0.90 |
| T base pop LIF: final − current | -0.19 / +1.16 / -0.78 | +0.06 | no clear difference | 0.94 |
| T (나) pop LIF: final − current | -0.27 / -0.69 / -1.05 | -0.67 | worse | 0.92 |
| T base pop f-LIF: final − current | -0.54 / +0.00 / +0.44 | -0.03 | no clear difference | 0.95 |
| T (나) pop f-LIF: final − current | +0.14 / +0.11 / -0.32 | -0.02 | no clear difference | 0.97 |
| S1 pop LIF: (나) − base, final | -0.36 / -1.40 / -0.21 | -0.66 | worse | 0.61 |
| S1 pop f-LIF: (나) − base, final | +0.15 / -0.50 / -1.26 | -0.54 | worse | 0.65 |
| S2 pop LIF: base − Spikformer(DE), final | +0.45 / +1.02 / -0.62 | +0.29 | no clear difference | 1.31 |
| S2 pop LIF: (나) − Spikformer(DE), final | +0.09 / -0.38 / -0.83 | -0.37 | no clear difference | 0.79 |
| S2 pop f-LIF: base − Spikformer(DE), final | +0.29 / +0.56 / +0.28 | +0.37 | no clear difference | 1.45 |
| S2 pop f-LIF: (나) − Spikformer(DE), final | +0.44 / +0.05 / -0.98 | -0.16 | no clear difference | 0.94 |
| (reference) pop LIF: (나) − base, current | -0.28 / +0.44 / +0.07 | +0.08 | no clear difference | 0.62 |
| (reference) pop LIF: base − Spikformer(DE), current | +0.03 / -0.33 / +0.85 | +0.18 | no clear difference | 1.25 |
| (reference) pop LIF: (나) − Spikformer(DE), current | -0.24 / +0.12 / +0.92 | +0.26 | no clear difference | 0.77 |
| (reference) pop f-LIF: (나) − base, current | -0.53 / -0.61 / -0.49 | -0.54 | worse | 0.63 |
| (reference) pop f-LIF: base − Spikformer(DE), current | +0.22 / +0.36 / +0.52 | +0.37 | no clear difference | 1.38 |
| (reference) pop f-LIF: (나) − Spikformer(DE), current | -0.31 / -0.25 / +0.03 | -0.18 | no clear difference | 0.87 |

## conclusions (rules of the pre-registration)

| comparison | task | pop LIF | pop f-LIF | conclusion over the two neurons |
|---|---|---|---|---|
| S1 (나) − base | ETT | no clear difference | no clear difference | no clear difference |
| S1 (나) − base | MIT-BIH | worse | worse | worse |
| S2 base − Spikformer(DE) | ETT | no clear difference | no clear difference | no clear difference |
| S2 base − Spikformer(DE) | MIT-BIH | no clear difference | no clear difference | no clear difference |
| overall: base better than Spikformer(DE) | both | weak (direction only) | weak (direction only) | — |
| S2 (나) − Spikformer(DE) | ETT | no clear difference | no clear difference | no clear difference |
| S2 (나) − Spikformer(DE) | MIT-BIH | no clear difference | no clear difference | no clear difference |
| overall: (나) better than Spikformer(DE) | both | not supported | not supported | — |

**Recommendation (pre-registered rule; the structure decision is the user's):** keep the base: S1 has a "worse". S1 verdicts (ETT pop LIF, ETT pop f-LIF, MIT-BIH pop LIF, MIT-BIH pop f-LIF): no clear difference, no clear difference, worse, worse.

## report only

| model | setting | ETT energy µJ / Spikformer(DE) final | MIT-BIH energy µJ / Spikformer(DE) final | ETT rho_hippo | ETT rho_neo | MIT-BIH rho_hippo | MIT-BIH rho_neo | MIT-BIH accuracy % | ETT best at epoch 0 (of 12) | ETT epochs (min–max) | ETT run time s (min–max) | MIT-BIH run time s (min–max) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Spikformer(DE) | current | 10.65 / 1.00 | 9.18 / 1.11 | 0.0610 | — | 0.1075 | — | 98.32 | 2 | 11–17 | 262–1145 | 544–613 |
| Spikformer(DE) | final | 10.67 / 1.00 | 8.27 / 1.00 | 0.0631 | — | 0.1204 | — | 98.29 | 3 | 11–15 | 324–2170 | 517–525 |
| base pop LIF | current | 13.20 / 1.24 | 11.45 / 1.38 | 0.0600 | 0.0347 | 0.1271 | 0.0785 | 98.40 | 2 | 11–16 | 448–2202 | 840–1243 |
| base pop LIF | final | 13.26 / 1.24 | 10.81 / 1.31 | 0.0610 | 0.0340 | 0.1212 | 0.0724 | 98.38 | 1 | 11–15 | 567–5435 | 797–1009 |
| (나) pop LIF | current | 8.48 / 0.79 | 7.10 / 0.86 | 0.0816 | 0.0469 | 0.1618 | 0.1229 | 98.41 | 4 | 11–15 | 391–1681 | 582–871 |
| (나) pop LIF | final | 8.44 / 0.79 | 6.57 / 0.79 | 0.0810 | 0.0478 | 0.1174 | 0.0854 | 98.27 | 1 | 11–15 | 401–5121 | 497–643 |
| base pop f-LIF | current | 16.00 / 1.50 | 12.65 / 1.53 | 0.0597 | 0.0346 | 0.1253 | 0.0955 | 98.42 | 2 | 11–17 | 459–2600 | 835–1087 |
| base pop f-LIF | final | 16.05 / 1.50 | 12.03 / 1.45 | 0.0599 | 0.0342 | 0.1157 | 0.0838 | 98.41 | 2 | 11–17 | 598–2600 | 1070–1301 |
| (나) pop f-LIF | current | 11.35 / 1.06 | 8.02 / 0.97 | 0.0847 | 0.0495 | 0.1492 | 0.1100 | 98.29 | 4 | 11–16 | 353–1950 | 578–783 |
| (나) pop f-LIF | final | 11.27 / 1.05 | 7.81 / 0.94 | 0.0814 | 0.0466 | 0.1305 | 0.0929 | 98.29 | 3 | 11–15 | 385–3923 | 723–869 |

### ETTh2: validation score of every combination, seed 42 (s = mean over H96 / H720 of val / val(current) − 1, %; lower is better; **selected**)

| lr | batch | warmup | Spikformer(DE) | base pop LIF | (나) pop LIF | base pop f-LIF | (나) pop f-LIF |
|---|---|---|---|---|---|---|---|
| 0.00025 | 32 | none | -0.093 | -0.079 | +0.014 | -0.129 | -0.019 |
| 0.00025 | 32 | 1 | **-0.477** | -0.142 | -0.113 | **-0.243** | +0.015 |
| 0.00025 | 64 | none | +0.239 | -0.064 | +0.023 | +0.127 | +0.002 |
| 0.00025 | 64 | 1 | +0.172 | +0.115 | -0.066 | +0.291 | -0.010 |
| 0.00025 | 128 | none | +0.211 | +0.394 | +0.212 | +0.305 | +0.225 |
| 0.00025 | 128 | 1 | +0.312 | +0.198 | +0.163 | +0.292 | +0.162 |
| 0.0005 | 32 | none | +0.334 | +0.028 | +0.070 | +0.153 | +0.177 |
| 0.0005 | 32 | 1 | -0.005 | -0.126 | +0.201 | +0.199 | +0.026 |
| 0.0005 | 64 | none | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 | (current)
| 0.0005 | 64 | 1 | -0.108 | **-0.376** | +0.038 | -0.011 | +0.035 |
| 0.0005 | 128 | none | +0.184 | +0.156 | +0.125 | +0.146 | +0.206 |
| 0.0005 | 128 | 1 | +0.085 | +0.095 | +0.250 | +0.401 | +0.225 |
| 0.001 | 32 | none | +0.289 | +0.285 | +0.295 | +0.304 | +0.343 |
| 0.001 | 32 | 1 | +0.202 | +0.401 | +0.190 | +0.617 | +0.380 |
| 0.001 | 64 | none | +0.232 | -0.059 | +0.219 | +0.119 | **-0.062** |
| 0.001 | 64 | 1 | +0.237 | +0.228 | +0.069 | +0.039 | +0.071 |
| 0.001 | 128 | none | +0.022 | +0.090 | **-0.128** | +0.174 | +0.164 |
| 0.001 | 128 | 1 | +0.209 | -0.026 | -0.120 | +0.176 | +0.107 |

### ETTm2: validation score of every combination, seed 42 (s = mean over H96 / H720 of val / val(current) − 1, %; lower is better; **selected**)

| lr | batch | warmup | Spikformer(DE) | base pop LIF | (나) pop LIF | base pop f-LIF | (나) pop f-LIF |
|---|---|---|---|---|---|---|---|
| 0.00025 | 32 | none | -0.008 | +0.016 | -0.096 | +0.007 | **-0.150** |
| 0.00025 | 32 | 1 | +0.029 | +0.046 | -0.091 | +0.020 | -0.079 |
| 0.00025 | 64 | none | +0.120 | +0.107 | +0.025 | +0.064 | -0.007 |
| 0.00025 | 64 | 1 | +0.206 | +0.296 | -0.031 | +0.059 | -0.040 |
| 0.00025 | 128 | none | +0.101 | +0.158 | -0.048 | +0.147 | -0.021 |
| 0.00025 | 128 | 1 | +0.111 | +0.249 | -0.008 | +0.167 | +0.019 |
| 0.0005 | 32 | none | **-0.059** | -0.014 | -0.116 | +0.039 | -0.074 |
| 0.0005 | 32 | 1 | +0.138 | **-0.025** | **-0.139** | +0.049 | -0.103 |
| 0.0005 | 64 | none | +0.000 | +0.000 | +0.000 | **+0.000** | +0.000 | (current)
| 0.0005 | 64 | 1 | +0.121 | +0.230 | -0.006 | +0.148 | -0.065 |
| 0.0005 | 128 | none | +0.165 | +0.214 | -0.090 | +0.175 | +0.104 |
| 0.0005 | 128 | 1 | +0.193 | +0.103 | +0.056 | +0.189 | +0.116 |
| 0.001 | 32 | none | +0.356 | +0.218 | +0.108 | +0.210 | +0.255 |
| 0.001 | 32 | 1 | +0.031 | +0.100 | +0.036 | +0.261 | +0.149 |
| 0.001 | 64 | none | +0.195 | +0.136 | -0.025 | +0.055 | +0.094 |
| 0.001 | 64 | 1 | +0.179 | -0.019 | +0.025 | +0.094 | +0.039 |
| 0.001 | 128 | none | +0.248 | +0.381 | -0.097 | +0.280 | +0.130 |
| 0.001 | 128 | 1 | +0.204 | +0.229 | +0.163 | +0.224 | +0.261 |

### MIT-BIH: last-epoch validation macro-F1 % of every combination, seed 42, VAL=1 (higher is better; **selected**); in brackets the test macro-F1 % of the same run (not used)

| lr | batch | warmup epochs | Spikformer(DE) | base pop LIF | (나) pop LIF | base pop f-LIF | (나) pop f-LIF |
|---|---|---|---|---|---|---|---|
| 0.005 | 256 | 6 | 91.84 [90.55] | 91.73 [90.75] | 92.41 [90.70] | 91.99 [90.83] | 91.84 [90.63] |
| 0.005 | 256 | 0 | 92.21 [90.26] | 91.90 [91.34] | 91.26 [91.12] | **92.64 [90.80]** | **92.42 [90.89]** |
| 0.005 | 512 | 6 | 92.07 [90.55] | 91.82 [91.28] | 91.71 [90.32] | 92.30 [90.62] | 91.95 [90.35] |
| 0.005 | 512 | 0 | 91.03 [90.25] | 91.76 [91.01] | 91.59 [91.00] | 92.12 [90.83] | 92.13 [90.60] |
| 0.005 | 1024 | 6 | 91.33 [90.34] | 91.68 [90.53] | 91.24 [90.42] | 91.79 [90.50] | 90.92 [90.12] |
| 0.005 | 1024 | 0 | 90.94 [90.07] | 91.03 [90.59] | 91.27 [90.78] | 91.19 [90.39] | 91.40 [89.83] |
| 0.01 | 256 | 6 | 90.87 [90.81] | 91.96 [90.99] | 91.25 [91.36] | 91.58 [90.99] | 92.16 [90.83] |
| 0.01 | 256 | 0 | 91.61 [90.63] | 91.18 [91.92] | 91.89 [90.80] | 92.20 [90.99] | 91.90 [90.82] |
| 0.01 | 512 | 6 | 91.36 [90.47] | 91.95 [90.71] | 91.59 [90.67] | 91.60 [91.08] | 91.76 [90.68] | (current)
| 0.01 | 512 | 0 | 91.83 [90.40] | **92.24 [90.77]** | 91.97 [90.34] | 92.02 [91.30] | 91.79 [89.78] |
| 0.01 | 1024 | 6 | 91.64 [90.63] | 91.87 [90.97] | 91.18 [90.61] | 90.84 [90.60] | 91.07 [90.41] |
| 0.01 | 1024 | 0 | 91.35 [89.91] | 91.27 [90.44] | 91.65 [90.56] | 91.69 [90.55] | 90.91 [89.99] |
| 0.02 | 256 | 6 | **92.61 [90.64]** | 90.73 [90.71] | 92.01 [90.65] | 91.97 [90.63] | 91.39 [90.29] |
| 0.02 | 256 | 0 | 91.20 [90.41] | 91.21 [90.64] | 91.83 [89.54] | 91.62 [91.01] | 91.04 [89.92] |
| 0.02 | 512 | 6 | 91.21 [91.05] | 92.01 [90.71] | 91.58 [90.28] | 92.14 [90.30] | 91.06 [90.50] |
| 0.02 | 512 | 0 | 92.06 [90.29] | 91.60 [90.49] | 91.78 [90.28] | 91.78 [90.57] | 91.60 [90.32] |
| 0.02 | 1024 | 6 | 91.06 [90.68] | 91.32 [90.37] | 91.12 [90.97] | 91.92 [90.98] | 91.23 [90.03] |
| 0.02 | 1024 | 0 | 91.73 [90.69] | 91.87 [90.80] | **92.51 [90.68]** | 91.37 [90.84] | 91.25 [89.57] |
