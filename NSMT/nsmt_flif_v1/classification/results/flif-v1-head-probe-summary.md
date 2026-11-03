# Head readout diagnostic (frozen query-form bodies, new linear readouts) — summary

See the module docstring and head_probe.py for sources, probes and rules. Probes R_k are compared with the R0 probe of the same checkpoint.

## MIT-BIH (test macro-F1 %, 80 / 20 --mitbih_val checkpoints, lr 0.01)

| model | readout | dim | seed 42 | seed 7 | seed 2026 | mean | lam (per seed) |
|---|---|---|---|---|---|---|---|
| query pop LIF | head  | 384 | 91.71 | 91.81 | 90.98 | 91.50 | trained head |
| query pop LIF | R0 current head input | 384 | 91.81 | 91.26 | 91.21 | 91.42 | 0.1 / 0.0001 / 0.1 |
| query pop LIF | R1 per token | 4224 | 91.61 | 91.88 | 91.22 | 91.57 | 0.0001 / 0.01 / 0.1 |
| query pop LIF | R2 per T_h step | 1536 | 91.82 | 91.30 | 91.12 | 91.41 | 0.001 / 1e-06 / 0.001 |
| query pop LIF | R3 + embedding spikes | 768 | 91.75 | 91.76 | 91.25 | 91.58 | 0.01 / 0.01 / 0.1 |
| query pop LIF | R4 + Neocortex output | 768 | 91.80 | 91.49 | 91.35 | 91.55 | 0.1 / 0.1 / 0.1 |
| query pop LIF | R5 + membrane potentials | 1152 | 91.27 | 91.35 | 90.49 | 91.04 | 0.01 / 0.0001 / 0.001 |
| query pop LIF | R6 raw input | 187 | 68.77 | 68.77 | 68.77 | 68.77 | 0.0001 / 0.0001 / 0.0001 |
| query pop LIF | R7 all sources per token | 21120 | 91.58 | 91.40 | 90.89 | 91.29 | 0.0001 / 0.01 / 0.01 |
| query pop f-LIF | head  | 384 | 91.75 | 91.20 | 91.62 | 91.52 | trained head |
| query pop f-LIF | R0 current head input | 384 | 91.34 | 91.31 | 91.77 | 91.47 | 0.01 / 0.0001 / 0.0001 |
| query pop f-LIF | R1 per token | 4224 | 91.42 | 91.67 | 91.65 | 91.58 | 0.01 / 0.01 / 0.001 |
| query pop f-LIF | R2 per T_h step | 1536 | 91.33 | 91.57 | 91.65 | 91.52 | 1e-06 / 1e-06 / 0.0001 |
| query pop f-LIF | R3 + embedding spikes | 768 | 91.37 | 91.52 | 91.62 | 91.50 | 0.01 / 0.0001 / 0.0001 |
| query pop f-LIF | R4 + Neocortex output | 768 | 91.43 | 91.40 | 91.80 | 91.54 | 0.01 / 0.0001 / 0.001 |
| query pop f-LIF | R5 + membrane potentials | 1152 | 91.08 | 91.02 | 91.53 | 91.21 | 0.01 / 1e-05 / 0.001 |
| query pop f-LIF | R6 raw input | 187 | 68.77 | 68.77 | 68.77 | 68.77 | 0.0001 / 0.0001 / 0.0001 |
| query pop f-LIF | R7 all sources per token | 21120 | 91.41 | 91.60 | 91.65 | 91.56 | 0.01 / 0.01 / 0.001 |

(* = lam at the cap of the grid.)

### verdicts: R_k probe − R0 probe (%p)

| readout | model | per seed (42 / 7 / 2026) | mean | > 0 | verdict |
|---|---|---|---|---|---|
| R1 per token | query pop LIF | -0.20 / +0.62 / +0.01 | +0.15 | 2/3 | no clear difference |
| R1 per token | query pop f-LIF | +0.08 / +0.36 / -0.11 | +0.11 | 2/3 | no clear difference |
| R2 per T_h step | query pop LIF | +0.02 / +0.04 / -0.09 | -0.01 | 2/3 | no clear difference |
| R2 per T_h step | query pop f-LIF | -0.01 / +0.27 / -0.11 | +0.05 | 1/3 | no clear difference |
| R3 + embedding spikes | query pop LIF | -0.06 / +0.50 / +0.04 | +0.16 | 2/3 | no clear difference |
| R3 + embedding spikes | query pop f-LIF | +0.03 / +0.21 / -0.15 | +0.03 | 2/3 | no clear difference |
| R4 + Neocortex output | query pop LIF | -0.01 / +0.24 / +0.14 | +0.12 | 2/3 | no clear difference |
| R4 + Neocortex output | query pop f-LIF | +0.09 / +0.10 / +0.03 | +0.07 | 3/3 | no clear difference |
| R5 + membrane potentials | query pop LIF | -0.54 / +0.09 / -0.72 | -0.39 | 1/3 | no clear difference |
| R5 + membrane potentials | query pop f-LIF | -0.26 / -0.29 / -0.23 | -0.26 | 0/3 | no clear difference |

### report only (%p, mean of 3 seeds)

| model | R0 probe − trained head | R6 − R0 probe | R6 − trained head | R7 − R0 probe |
|---|---|---|---|---|
| query pop LIF | -0.08 | -22.65 | -22.73 | -0.13 |
| query pop f-LIF | -0.05 | -22.70 | -22.75 | +0.08 |

L-BFGS fits at the selected lam that hit the 1000-iteration limit (not converged): 0

Per-class test F1 (%, mean of 3 seeds; classes N / S / V / F / Q):

| model | readout | N | S | V | F | Q |
|---|---|---|---|---|---|---|
| query pop LIF | head | 99.17 | 84.49 | 95.72 | 79.33 | 98.80 |
| query pop LIF | R0 | 99.17 | 84.39 | 95.49 | 79.28 | 98.78 |
| query pop LIF | R1 | 99.17 | 84.67 | 95.60 | 79.64 | 98.78 |
| query pop LIF | R2 | 99.18 | 84.46 | 95.72 | 78.93 | 98.78 |
| query pop LIF | R3 | 99.16 | 84.63 | 95.62 | 79.78 | 98.73 |
| query pop LIF | R4 | 99.16 | 84.49 | 95.55 | 79.77 | 98.77 |
| query pop LIF | R5 | 99.14 | 84.00 | 95.52 | 77.85 | 98.67 |
| query pop LIF | R6 | 94.62 | 60.51 | 51.98 | 45.56 | 91.17 |
| query pop LIF | R7 | 99.15 | 84.11 | 95.67 | 78.77 | 98.75 |
| query pop f-LIF | head | 99.19 | 84.37 | 96.06 | 79.14 | 98.86 |
| query pop f-LIF | R0 | 99.18 | 83.91 | 95.90 | 79.55 | 98.84 |
| query pop f-LIF | R1 | 99.19 | 84.34 | 96.07 | 79.38 | 98.93 |
| query pop f-LIF | R2 | 99.18 | 83.87 | 95.98 | 79.77 | 98.79 |
| query pop f-LIF | R3 | 99.18 | 83.96 | 96.00 | 79.61 | 98.78 |
| query pop f-LIF | R4 | 99.17 | 83.84 | 95.96 | 79.94 | 98.80 |
| query pop f-LIF | R5 | 99.15 | 83.65 | 95.91 | 78.58 | 98.76 |
| query pop f-LIF | R6 | 94.62 | 60.51 | 51.98 | 45.56 | 91.17 |
| query pop f-LIF | R7 | 99.19 | 84.09 | 95.90 | 79.78 | 98.83 |

## ETT (test MSE, scaled data, best-validation checkpoints)

### ETTh2 H96

| model | readout | dim | seed 42 | seed 7 | seed 2026 | mean | lam (per seed) |
|---|---|---|---|---|---|---|---|
| query pop LIF | head  |  | 0.286700 | 0.295619 | 0.291308 | 0.291209 | trained head |
| query pop LIF | R0 current head input | 1536 | 0.289994 | 0.289706 | 0.291090 | 0.290264 | 10 / 10 / 10 |
| query pop LIF | R2 per T_h step | 6144 | 0.291058 | 0.290001 | 0.291248 | 0.290769 | 10 / 10 / 10 |
| query pop LIF | R3 + embedding spikes | 3072 | 0.289310 | 0.289001 | 0.290214 | 0.289508 | 10 / 10 / 10 |
| query pop LIF | R4 + Neocortex output | 3072 | 0.290160 | 0.289708 | 0.290594 | 0.290154 | 10 / 10 / 10 |
| query pop LIF | R5 + membrane potentials | 4608 | 0.291608 | 0.289788 | 0.293399 | 0.291598 | 10 / 10 / 10 |
| query pop LIF | R6 raw input | 96 | 0.287599 | 0.287599 | 0.287599 | 0.287599 | 0.1 / 0.1 / 0.1 |
| query pop LIF | R7 all sources per token | 7680 | 0.291181 | 0.289324 | 0.291928 | 0.290811 | 10 / 10 / 10 |
| query pop f-LIF | head  |  | 0.283844 | 0.293645 | 0.289657 | 0.289049 | trained head |
| query pop f-LIF | R0 current head input | 1536 | 0.286347 | 0.289225 | 0.287825 | 0.287799 | 10 / 10 / 10 |
| query pop f-LIF | R2 per T_h step | 6144 | 0.287298 | 0.290851 | 0.288205 | 0.288785 | 10 / 10 / 10 |
| query pop f-LIF | R3 + embedding spikes | 3072 | 0.285834 | 0.288880 | 0.287213 | 0.287309 | 10 / 10 / 10 |
| query pop f-LIF | R4 + Neocortex output | 3072 | 0.286811 | 0.290453 | 0.287771 | 0.288345 | 10 / 10 / 10 |
| query pop f-LIF | R5 + membrane potentials | 4608 | 0.287530 | 0.290109 | 0.288815 | 0.288818 | 10 / 10 / 10 |
| query pop f-LIF | R6 raw input | 96 | 0.287599 | 0.287599 | 0.287599 | 0.287599 | 0.1 / 0.1 / 0.1 |
| query pop f-LIF | R7 all sources per token | 7680 | 0.286862 | 0.289489 | 0.287348 | 0.287900 | 10 / 10 / 10 |

### ETTh2 H720

| model | readout | dim | seed 42 | seed 7 | seed 2026 | mean | lam (per seed) |
|---|---|---|---|---|---|---|---|
| query pop LIF | head  |  | 0.414568 | 0.414103 | 0.411962 | 0.413544 | trained head |
| query pop LIF | R0 current head input | 1536 | 0.413812 | 0.415141 | 0.415499 | 0.414817 | 10 / 10 / 10 |
| query pop LIF | R2 per T_h step | 6144 | 0.413212 | 0.415290 | 0.415428 | 0.414643 | 10 / 10 / 10 |
| query pop LIF | R3 + embedding spikes | 3072 | 0.412927 | 0.414281 | 0.414912 | 0.414040 | 10 / 10 / 10 |
| query pop LIF | R4 + Neocortex output | 3072 | 0.415645 | 0.420258 | 0.417672 | 0.417858 | 10 / 10 / 10 |
| query pop LIF | R5 + membrane potentials | 4608 | 0.411893 | 0.414150 | 0.414603 | 0.413548 | 10 / 10 / 10 |
| query pop LIF | R6 raw input | 96 | 0.419389 | 0.419389 | 0.419389 | 0.419389 | 0.1 / 0.1 / 0.1 |
| query pop LIF | R7 all sources per token | 7680 | 0.419064 | 0.421974 | 0.418698 | 0.419912 | 100 / 100 / 100 |
| query pop f-LIF | head  |  | 0.415474 | 0.413353 | 0.411801 | 0.413543 | trained head |
| query pop f-LIF | R0 current head input | 1536 | 0.413540 | 0.414460 | 0.414328 | 0.414110 | 10 / 10 / 10 |
| query pop f-LIF | R2 per T_h step | 6144 | 0.413811 | 0.415734 | 0.414155 | 0.414566 | 10 / 10 / 10 |
| query pop f-LIF | R3 + embedding spikes | 3072 | 0.412160 | 0.413566 | 0.412857 | 0.412861 | 10 / 10 / 10 |
| query pop f-LIF | R4 + Neocortex output | 3072 | 0.415130 | 0.416418 | 0.416008 | 0.415852 | 10 / 10 / 10 |
| query pop f-LIF | R5 + membrane potentials | 4608 | 0.421494 | 0.411317 | 0.413211 | 0.415340 | 100 / 10 / 10 |
| query pop f-LIF | R6 raw input | 96 | 0.419389 | 0.419389 | 0.419389 | 0.419389 | 0.1 / 0.1 / 0.1 |
| query pop f-LIF | R7 all sources per token | 7680 | 0.417089 | 0.418524 | 0.418716 | 0.418110 | 100 / 100 / 100 |

### ETTm2 H96

| model | readout | dim | seed 42 | seed 7 | seed 2026 | mean | lam (per seed) |
|---|---|---|---|---|---|---|---|
| query pop LIF | head  |  | 0.176038 | 0.176194 | 0.175893 | 0.176042 | trained head |
| query pop LIF | R0 current head input | 1536 | 0.174874 | 0.174753 | 0.175075 | 0.174901 | 1 / 1 / 10 |
| query pop LIF | R2 per T_h step | 6144 | 0.173382 | 0.173078 | 0.172909 | 0.173123 | 10 / 10 / 10 |
| query pop LIF | R3 + embedding spikes | 3072 | 0.174583 | 0.174002 | 0.173935 | 0.174173 | 10 / 10 / 10 |
| query pop LIF | R4 + Neocortex output | 3072 | 0.175253 | 0.174698 | 0.174482 | 0.174811 | 1 / 10 / 10 |
| query pop LIF | R5 + membrane potentials | 4608 | 0.175228 | 0.174251 | 0.174195 | 0.174558 | 10 / 10 / 10 |
| query pop LIF | R6 raw input | 96 | 0.181802 | 0.181802 | 0.181802 | 0.181802 | 0.1 / 0.1 / 0.1 |
| query pop LIF | R7 all sources per token | 7680 | 0.174736 | 0.173842 | 0.173843 | 0.174140 | 10 / 10 / 10 |
| query pop f-LIF | head  |  | 0.176426 | 0.175629 | 0.177526 | 0.176527 | trained head |
| query pop f-LIF | R0 current head input | 1536 | 0.176104 | 0.175371 | 0.176764 | 0.176080 | 1 / 10 / 1 |
| query pop f-LIF | R2 per T_h step | 6144 | 0.174335 | 0.173452 | 0.174540 | 0.174109 | 10 / 10 / 10 |
| query pop f-LIF | R3 + embedding spikes | 3072 | 0.175152 | 0.174277 | 0.175060 | 0.174830 | 10 / 10 / 10 |
| query pop f-LIF | R4 + Neocortex output | 3072 | 0.175437 | 0.174389 | 0.175969 | 0.175265 | 10 / 10 / 10 |
| query pop f-LIF | R5 + membrane potentials | 4608 | 0.175548 | 0.174608 | 0.175602 | 0.175253 | 10 / 10 / 10 |
| query pop f-LIF | R6 raw input | 96 | 0.181802 | 0.181802 | 0.181802 | 0.181802 | 0.1 / 0.1 / 0.1 |
| query pop f-LIF | R7 all sources per token | 7680 | 0.174910 | 0.173904 | 0.175249 | 0.174688 | 10 / 10 / 10 |

### ETTm2 H720

| model | readout | dim | seed 42 | seed 7 | seed 2026 | mean | lam (per seed) |
|---|---|---|---|---|---|---|---|
| query pop LIF | head  |  | 0.400544 | 0.402993 | 0.404856 | 0.402798 | trained head |
| query pop LIF | R0 current head input | 1536 | 0.398195 | 0.398785 | 0.400711 | 0.399230 | 10 / 10 / 10 |
| query pop LIF | R2 per T_h step | 6144 | 0.397721 | 0.399272 | 0.402583 | 0.399859 | 10 / 10 / 10 |
| query pop LIF | R3 + embedding spikes | 3072 | 0.397723 | 0.398037 | 0.400132 | 0.398631 | 10 / 10 / 10 |
| query pop LIF | R4 + Neocortex output | 3072 | 0.397379 | 0.397775 | 0.400456 | 0.398536 | 10 / 10 / 10 |
| query pop LIF | R5 + membrane potentials | 4608 | 0.399464 | 0.399897 | 0.403029 | 0.400797 | 10 / 10 / 10 |
| query pop LIF | R6 raw input | 96 | 0.407078 | 0.407078 | 0.407078 | 0.407078 | 0.1 / 0.1 / 0.1 |
| query pop LIF | R7 all sources per token | 7680 | 0.399200 | 0.399315 | 0.403120 | 0.400545 | 10 / 10 / 10 |
| query pop f-LIF | head  |  | 0.400702 | 0.401670 | 0.403459 | 0.401944 | trained head |
| query pop f-LIF | R0 current head input | 1536 | 0.397896 | 0.397091 | 0.398759 | 0.397915 | 10 / 10 / 10 |
| query pop f-LIF | R2 per T_h step | 6144 | 0.397595 | 0.397674 | 0.399512 | 0.398260 | 10 / 10 / 10 |
| query pop f-LIF | R3 + embedding spikes | 3072 | 0.397448 | 0.396626 | 0.398243 | 0.397439 | 10 / 10 / 10 |
| query pop f-LIF | R4 + Neocortex output | 3072 | 0.396787 | 0.396132 | 0.397392 | 0.396770 | 10 / 10 / 10 |
| query pop f-LIF | R5 + membrane potentials | 4608 | 0.397916 | 0.397958 | 0.399771 | 0.398548 | 10 / 10 / 10 |
| query pop f-LIF | R6 raw input | 96 | 0.407078 | 0.407078 | 0.407078 | 0.407078 | 0.1 / 0.1 / 0.1 |
| query pop f-LIF | R7 all sources per token | 7680 | 0.397443 | 0.397981 | 0.399766 | 0.398397 | 10 / 10 / 10 |

### verdicts: R_k probe / R0 probe − 1 (%)

| readout | model | pooled 12 pairs | < 0 | verdict | per cell (mean, verdict) |
|---|---|---|---|---|---|
| R2 per T_h step | query pop LIF | -0.18 | 6/12 | no clear difference | ETTh2 H96 +0.17 no clear difference; ETTh2 H720 -0.04 no clear difference; ETTm2 H96 -1.02 better; ETTm2 H720 +0.16 no clear difference |
| R2 per T_h step | query pop f-LIF | -0.14 | 5/12 | no clear difference | ETTh2 H96 +0.34 no clear difference; ETTh2 H720 +0.11 no clear difference; ETTm2 H96 -1.12 better; ETTm2 H720 +0.09 no clear difference |
| R3 + embedding spikes | query pop LIF | -0.25 | 12/12 | no clear difference | ETTh2 H96 -0.26 no clear difference; ETTh2 H720 -0.19 no clear difference; ETTm2 H96 -0.42 no clear difference; ETTm2 H720 -0.15 no clear difference |
| R3 + embedding spikes | query pop f-LIF | -0.33 | 12/12 | no clear difference | ETTh2 H96 -0.17 no clear difference; ETTh2 H720 -0.30 no clear difference; ETTm2 H96 -0.71 no clear difference; ETTm2 H720 -0.12 no clear difference |
| R4 + Neocortex output | query pop LIF | +0.12 | 6/12 | no clear difference | ETTh2 H96 -0.04 no clear difference; ETTh2 H720 +0.73 no clear difference; ETTm2 H96 -0.05 no clear difference; ETTm2 H720 -0.17 no clear difference |
| R4 + Neocortex output | query pop f-LIF | -0.04 | 7/12 | no clear difference | ETTh2 H96 +0.19 no clear difference; ETTh2 H720 +0.42 no clear difference; ETTm2 H96 -0.46 no clear difference; ETTm2 H720 -0.29 no clear difference |
| R5 + membrane potentials | query pop LIF | +0.09 | 5/12 | no clear difference | ETTh2 H96 +0.46 no clear difference; ETTh2 H720 -0.31 no clear difference; ETTm2 H96 -0.20 no clear difference; ETTm2 H720 +0.39 no clear difference |
| R5 + membrane potentials | query pop f-LIF | +0.09 | 5/12 | no clear difference | ETTh2 H96 +0.35 no clear difference; ETTh2 H720 +0.30 no clear difference; ETTm2 H96 -0.47 no clear difference; ETTm2 H720 +0.16 no clear difference |

### report only (%, mean over 3 seeds per cell)

| model | cell | R0 probe / trained head − 1 | R6 / R0 probe − 1 | R6 / trained head − 1 | R7 / R0 probe − 1 |
|---|---|---|---|---|---|
| query pop LIF | ETTh2 H96 | -0.31 | -0.92 | -1.22 | +0.19 |
| query pop LIF | ETTh2 H720 | +0.31 | +1.10 | +1.41 | +1.23 |
| query pop LIF | ETTm2 H96 | -0.65 | +3.95 | +3.27 | -0.43 |
| query pop LIF | ETTm2 H720 | -0.88 | +1.97 | +1.06 | +0.33 |
| query pop f-LIF | ETTh2 H96 | -0.42 | -0.07 | -0.48 | +0.04 |
| query pop f-LIF | ETTh2 H720 | +0.14 | +1.27 | +1.42 | +0.97 |
| query pop f-LIF | ETTm2 H96 | -0.25 | +3.25 | +2.99 | -0.79 |
| query pop f-LIF | ETTm2 H720 | -1.00 | +2.30 | +1.28 | +0.12 |

## conclusions (rule of the pre-registration)

| task | readout | pop LIF | pop f-LIF | conclusion |
|---|---|---|---|---|
| MIT-BIH | R1 per token | no clear difference | no clear difference | no evidence |
| MIT-BIH | R2 per T_h step | no clear difference | no clear difference | no evidence |
| MIT-BIH | R3 + embedding spikes | no clear difference | no clear difference | no evidence |
| MIT-BIH | R4 + Neocortex output | no clear difference | no clear difference | no evidence |
| MIT-BIH | R5 + membrane potentials | no clear difference | no clear difference | no evidence |
| ETT | R2 per T_h step | no clear difference | no clear difference | no evidence |
| ETT | R3 + embedding spikes | no clear difference | no clear difference | no evidence |
| ETT | R4 + Neocortex output | no clear difference | no clear difference | no evidence |
| ETT | R5 + membrane potentials | no clear difference | no clear difference | no evidence |

## report: ETT test MSE of the trained models (mean of 3 seeds, best-validation weights)

| model | ETTh2 H96 | ETTh2 H720 | ETTm2 H96 | ETTm2 H720 | source |
|---|---|---|---|---|---|
| Spikformer-style | 0.2903 | 0.4223 | 0.1775 | 0.4034 | flif-v1-gen-ett |
| B form pop LIF | 0.2941 | 0.4192 | 0.1765 | 0.4025 | flif-v1-ett-dual |
| B form pop f-LIF | 0.2897 | 0.4193 | 0.1770 | 0.4055 | flif-v1-ett-dual |
| replace pop LIF | 0.2910 | 0.4101 | 0.1760 | 0.4003 | flif-v1-ett-dual |
| replace pop f-LIF | 0.2915 | 0.4170 | 0.1770 | 0.4023 | flif-v1-ett-dual |
| query pop LIF | 0.2912 | 0.4135 | 0.1760 | 0.4028 | flif-v1-ett-query + -m2 |
| query pop f-LIF | 0.2890 | 0.4135 | 0.1765 | 0.4019 | flif-v1-ett-query + -m2 |
| linear baseline (10-18) | 0.2915 | 0.4208 | 0.1829 | 0.4078 | flif-v1-s4/summary.md |
