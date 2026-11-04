# Head test H2 / H6 (query population, new heads on our models only) — summary

See the module docstring and docs/MODI_PROJECT_LOG.md 2026-11-04 for the pre-registered rules.

## H2 — attention pooling in front of the head (MIT-BIH, test macro-F1 %, last-epoch weights, full training file, lr 0.01)

| model | head | seed 42 | seed 7 | seed 2026 | mean | energy uJ (logged) | params |
|---|---|---|---|---|---|---|---|
| query pop LIF | mean (baseline) | 92.19 | 92.00 | 92.33 | 92.17 | 11.10 | 1230888 |
| query pop LIF | H2 attention pooling | 92.16 | 91.75 | 92.33 | 92.08 | 9.86 | 1231272 |
| query pop f-LIF | mean (baseline) | 92.04 | 91.84 | 92.18 | 92.02 | 12.26 | 1230888 |
| query pop f-LIF | H2 attention pooling | 92.11 | 91.94 | 91.92 | 91.99 | 11.17 | 1231272 |

### verdict: H2 − mean head (%p, paired by seed)

| model | per seed (42 / 7 / 2026) | mean | > 0 | verdict |
|---|---|---|---|---|
| query pop LIF | -0.03 / -0.26 / +0.00 | -0.09 | 1/3 | no clear difference |
| query pop f-LIF | +0.07 / +0.10 / -0.25 | -0.03 | 2/3 | no clear difference |

### report only

Per-class test F1 (%, mean of 3 seeds; N / S / V / F / Q):

| model | head | N | S | V | F | Q |
|---|---|---|---|---|---|---|
| query pop LIF | mean | 99.26 | 86.18 | 96.08 | 80.36 | 98.98 |
| query pop LIF | H2 | 99.24 | 85.56 | 95.81 | 80.71 | 99.08 |
| query pop f-LIF | mean | 99.23 | 85.70 | 96.14 | 79.97 | 99.04 |
| query pop f-LIF | H2 | 99.23 | 85.71 | 96.12 | 79.94 | 98.95 |

Spikformer-style (original head, flif-v1-gen-mitbih): mean 92.22; H2 − Spikformer-style: query pop LIF -0.14, query pop f-LIF -0.23 %p (body and head both differ: user decision 3)

Learned query |q| (starts at 0; the scores are q . x / sqrt(384)): query pop LIF seed 42 21.651; query pop LIF seed 7 20.675; query pop LIF seed 2026 20.792; query pop f-LIF seed 42 20.665; query pop f-LIF seed 7 20.551; query pop f-LIF seed 2026 19.684

## H6 — linear shortcut + small head (ETT, test MSE on scaled data, best-validation weights)

### ETTh2 H96

| model | head | seed 42 | seed 7 | seed 2026 | mean | val L1 (mean) | params |
|---|---|---|---|---|---|---|---|
| query pop LIF | flat (baseline) | 0.286700 | 0.295619 | 0.291308 | 0.291209 | 0.312312 | 175544 |
| query pop LIF | H6 | 0.286677 | 0.289330 | 0.282231 | 0.286079 | 0.310943 | 56248 |
| query pop LIF | H6 shortcut alone (report) | 0.296240 | 0.297351 | 0.296677 | 0.296756 |  |  |
| query pop f-LIF | flat (baseline) | 0.283844 | 0.293645 | 0.289657 | 0.289049 | 0.311733 | 175544 |
| query pop f-LIF | H6 | 0.284111 | 0.287894 | 0.284165 | 0.285390 | 0.310500 | 56248 |
| query pop f-LIF | H6 shortcut alone (report) | 0.296085 | 0.298032 | 0.296544 | 0.296887 |  |  |

### ETTh2 H720

| model | head | seed 42 | seed 7 | seed 2026 | mean | val L1 (mean) | params |
|---|---|---|---|---|---|---|---|
| query pop LIF | flat (baseline) | 0.414568 | 0.414103 | 0.411962 | 0.413544 | 0.536678 | 1134008 |
| query pop LIF | H6 | 0.415717 | 0.419999 | 0.415816 | 0.417177 | 0.537987 | 235960 |
| query pop LIF | H6 shortcut alone (report) | 0.422090 | 0.422555 | 0.420579 | 0.421741 |  |  |
| query pop f-LIF | flat (baseline) | 0.415474 | 0.413353 | 0.411801 | 0.413543 | 0.536754 | 1134008 |
| query pop f-LIF | H6 | 0.416244 | 0.412764 | 0.416874 | 0.415294 | 0.538209 | 235960 |
| query pop f-LIF | H6 shortcut alone (report) | 0.419007 | 0.422802 | 0.422378 | 0.421396 |  |  |

### ETTm2 H96

| model | head | seed 42 | seed 7 | seed 2026 | mean | val L1 (mean) | params |
|---|---|---|---|---|---|---|---|
| query pop LIF | flat (baseline) | 0.176038 | 0.176194 | 0.175893 | 0.176042 | 0.237016 | 184560 |
| query pop LIF | H6 | 0.175940 | 0.175827 | 0.176539 | 0.176102 | 0.237806 | 65264 |
| query pop LIF | H6 shortcut alone (report) | 0.192731 | 0.189735 | 0.193115 | 0.191860 |  |  |
| query pop f-LIF | flat (baseline) | 0.176426 | 0.175629 | 0.177526 | 0.176527 | 0.236709 | 184560 |
| query pop f-LIF | H6 | 0.178757 | 0.176420 | 0.178616 | 0.177931 | 0.237776 | 65264 |
| query pop f-LIF | H6 shortcut alone (report) | 0.192133 | 0.191807 | 0.187638 | 0.190526 |  |  |

### ETTm2 H720

| model | head | seed 42 | seed 7 | seed 2026 | mean | val L1 (mean) | params |
|---|---|---|---|---|---|---|---|
| query pop LIF | flat (baseline) | 0.400544 | 0.402993 | 0.404856 | 0.402798 | 0.361002 | 1143024 |
| query pop LIF | H6 | 0.402244 | 0.402242 | 0.402047 | 0.402178 | 0.360484 | 244976 |
| query pop LIF | H6 shortcut alone (report) | 0.409329 | 0.411731 | 0.409434 | 0.410165 |  |  |
| query pop f-LIF | flat (baseline) | 0.400702 | 0.401670 | 0.403459 | 0.401944 | 0.361032 | 1143024 |
| query pop f-LIF | H6 | 0.401517 | 0.401432 | 0.401091 | 0.401347 | 0.360607 | 244976 |
| query pop f-LIF | H6 shortcut alone (report) | 0.409444 | 0.409753 | 0.408223 | 0.409140 |  |  |

### verdict: H6 / flat − 1 (%, 12 pairs per model)

| model | pooled mean | < 0 | verdict | per cell (mean, verdict) |
|---|---|---|---|---|
| query pop LIF | -0.25 | 7/12 | no clear difference | ETTh2 H96 -1.75 better; ETTh2 H720 +0.88 no clear difference; ETTm2 H96 +0.03 no clear difference; ETTm2 H720 -0.15 no clear difference |
| query pop f-LIF | -0.05 | 5/12 | no clear difference | ETTh2 H96 -1.25 better; ETTh2 H720 +0.42 no clear difference; ETTm2 H96 +0.80 no clear difference; ETTm2 H720 -0.15 no clear difference |

### report only (mean of 3 seeds per cell)

| model | cell | H6 / linear baseline 10-18 − 1 (%) | linear baseline 10-18 | body term use: H6 / shortcut alone − 1 (%) | energy uJ: flat recount (spike-driven head) | flat logged | H6 logged |
|---|---|---|---|---|---|---|---|
| query pop LIF | ETTh2 H96 | -1.86 | 0.2915 | -3.60 | 11.27 | 15.73 | 11.97 |
| query pop LIF | ETTh2 H720 | -0.86 | 0.4208 | -1.08 | 12.92 | 46.61 | 17.69 |
| query pop LIF | ETTm2 H96 | -3.69 | 0.1829 | -8.21 | 13.23 | 17.72 | 13.73 |
| query pop LIF | ETTm2 H720 | -1.39 | 0.4078 | -1.95 | 14.36 | 48.18 | 19.11 |
| query pop f-LIF | ETTh2 H96 | -2.10 | 0.2915 | -3.87 | 13.54 | 18.02 | 14.19 |
| query pop f-LIF | ETTh2 H720 | -1.31 | 0.4208 | -1.45 | 15.16 | 48.85 | 19.93 |
| query pop f-LIF | ETTm2 H96 | -2.69 | 0.1829 | -6.60 | 16.53 | 21.02 | 17.07 |
| query pop f-LIF | ETTm2 H720 | -1.59 | 0.4078 | -1.90 | 17.74 | 51.58 | 22.52 |

Spikformer-style test MSE (original flat head, flif-v1-gen-ett; ETTh2 H96 / H720 / ETTm2 H96 / H720): 0.2903 / 0.4223 / 0.1775 / 0.4034 (body and head both differ from H6: user decision 3)

## conclusions (rule of the pre-registration)

| head | pop LIF | pop f-LIF | conclusion |
|---|---|---|---|
| H2 (MIT-BIH) | no clear difference | no clear difference | no clear difference |
| H6 (ETT) | no clear difference | no clear difference | no clear difference |
