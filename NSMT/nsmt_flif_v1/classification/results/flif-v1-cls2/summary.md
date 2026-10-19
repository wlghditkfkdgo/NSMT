# MIT-BIH suite `flif-v1-cls2` — summary (with reference runs from `flif-v1-cls`)

Test file: 32,833 beats (30% split). Last-epoch weights after 30 epochs (the ECG framework protocol). Numbers are mean ± standard deviation over seeds 42 / 7 / 2026, in %.

## Performance and energy

| condition | n | accuracy | macro-F1 | F1 S | F1 V | F1 F | F1 Q | synaptic µJ | total µJ (FFT / direct) | neocortex firing |
|---|---|---|---|---|---|---|---|---|---|---|
| ECG model (spikingjelly LIF) | 3 | 98.41 ± 0.03 | 91.58 ± 0.06 | 84.31 ± 0.46 | 95.65 ± 0.11 | 79.89 ± 0.71 | 98.89 ± 0.09 | 19.06 ± 0.18 | 23.86 ± 0.19 / 23.86 ± 0.19 | 0.0931 ± 0.0011 |
| spikeDE LIF, both | 3 | 98.39 ± 0.03 | 91.42 ± 0.29 | 83.77 ± 0.49 | 95.73 ± 0.06 | 79.72 ± 1.19 | 98.70 ± 0.11 | 18.36 ± 0.26 | 23.08 ± 0.25 / 23.08 ± 0.25 | 0.1142 ± 0.0053 |
| f-LIF, both | 3 | 98.35 ± 0.07 | 91.39 ± 0.42 | 83.11 ± 0.49 | 95.72 ± 0.10 | 80.18 ± 1.44 | 98.82 ± 0.06 | 18.57 ± 0.34 | 23.43 ± 0.37 / 23.67 ± 0.37 | 0.1056 ± 0.0079 |
| spikeDE LIF, MLP only | 3 | 98.38 ± 0.02 | 91.45 ± 0.23 | 83.69 ± 0.22 | 95.72 ± 0.10 | 79.90 ± 1.23 | 98.78 ± 0.10 | 18.96 ± 0.13 | 23.73 ± 0.13 / 23.73 ± 0.13 | 0.1136 ± 0.0034 |
| f-LIF, MLP only | 3 | 98.39 ± 0.04 | 91.42 ± 0.20 | 84.08 ± 0.49 | 95.60 ± 0.14 | 79.43 ± 0.55 | 98.82 ± 0.08 | 18.65 ± 0.03 | 23.50 ± 0.03 / 23.66 ± 0.03 | 0.1072 ± 0.0031 |
| no neocortex (ECG ablation) | 3 | 98.13 ± 0.07 | 90.40 ± 0.34 | 79.95 ± 0.70 | 95.40 ± 0.16 | 78.88 ± 1.20 | 98.75 ± 0.18 | 11.32 ± 0.03 | 13.03 ± 0.03 / 13.03 ± 0.03 | n/a |
| hippo-only (decision 7) | 3 | 98.43 ± 0.04 | 91.83 ± 0.23 | 84.01 ± 1.03 | 95.90 ± 0.23 | 81.23 ± 0.54 | 98.83 ± 0.10 | 21.41 ± 0.15 | 26.02 ± 0.16 / 26.02 ± 0.16 | n/a |
| population f-LIF, both | 3 | 98.39 ± 0.04 | 91.32 ± 0.04 | 83.46 ± 0.54 | 95.88 ± 0.24 | 79.17 ± 0.59 | 98.95 ± 0.02 | 18.32 ± 0.14 | 24.83 ± 0.16 / 25.21 ± 0.16 | 0.0793 ± 0.0100 |
| bidirectional spikeDE LIF | 3 | 98.44 ± 0.03 | 91.65 ± 0.20 | 84.28 ± 0.59 | 95.95 ± 0.29 | 80.06 ± 0.30 | 98.79 ± 0.06 | 19.49 ± 0.33 | 24.41 ± 0.36 / 24.41 ± 0.36 | 0.0850 ± 0.0041 |
| bidirectional f-LIF | 3 | 98.36 ± 0.06 | 91.23 ± 0.31 | 83.49 ± 0.80 | 95.73 ± 0.18 | 78.96 ± 0.94 | 98.82 ± 0.09 | 19.29 ± 0.09 | 24.41 ± 0.10 / 24.88 ± 0.10 | 0.0713 ± 0.0054 |
| bidirectional population f-LIF | 3 | 98.45 ± 0.04 | 91.70 ± 0.35 | 84.80 ± 0.40 | 95.93 ± 0.21 | 79.81 ± 1.49 | 98.78 ± 0.13 | 18.62 ± 0.13 | 27.09 ± 0.16 / 27.86 ± 0.16 | 0.0373 ± 0.0037 |

## Pre-registered comparisons (macro-F1, paired by seed, %p)

| question | A | B | A − B per seed (42 / 7 / 2026) | mean | verdict |
|---|---|---|---|---|---|
| Q1 f-LIF vs spikeDE LIF | f-LIF, both | spikeDE LIF, both | +0.19 / -0.04 / -0.22 | -0.02 | no clear difference |
| Q1 f-LIF vs spikeDE LIF | f-LIF, MLP only | spikeDE LIF, MLP only | -0.08 / +0.25 / -0.26 | -0.03 | no clear difference |
| Q2 f-LIF vs ECG model | f-LIF, both | ECG model (spikingjelly LIF) | +0.22 / -0.51 / -0.29 | -0.19 | no clear difference |
| Q2 f-LIF vs ECG model | f-LIF, MLP only | ECG model (spikingjelly LIF) | -0.05 / -0.10 / -0.33 | -0.16 | no clear difference |
| Q3 neocortex contribution | ECG model (spikingjelly LIF) | no neocortex (ECG ablation) | +0.86 / +1.29 / +1.40 | +1.18 | better |
| Q3 neocortex contribution | ECG model (spikingjelly LIF) | hippo-only (decision 7) | -0.42 / -0.25 / -0.07 | -0.25 | no clear difference |
| Q4 population f-LIF vs f-LIF | population f-LIF, both | f-LIF, both | -0.50 / +0.24 / +0.05 | -0.07 | no clear difference |
| Q5 bidirectional vs forward (f-LIF) | bidirectional f-LIF | f-LIF, both | -0.40 / +0.27 / -0.37 | -0.16 | no clear difference |
| Q5 bidirectional vs forward (spikeDE LIF) | bidirectional spikeDE LIF | spikeDE LIF, both | -0.25 / +0.72 / +0.23 | +0.24 | no clear difference |
| Q6 both vs f-LIF | bidirectional population f-LIF | f-LIF, both | +0.14 / +0.74 / +0.06 | +0.31 | no clear difference |
| Q6 both vs bidirectional f-LIF | bidirectional population f-LIF | bidirectional f-LIF | +0.53 / +0.47 / +0.43 | +0.48 | no clear difference |
| Q7 best stage-6 vs hippo-only | bidirectional population f-LIF | hippo-only (decision 7) | -0.06 / -0.01 / -0.30 | -0.12 | no clear difference |
| Q7 vs hippo-only | bidirectional f-LIF | hippo-only (decision 7) | -0.59 / -0.48 / -0.73 | -0.60 | worse |
| Q7 vs hippo-only | population f-LIF, both | hippo-only (decision 7) | -0.69 / -0.51 / -0.31 | -0.50 | worse |

Accuracy, same rule:

| A | B | A − B per seed | mean | verdict |
|---|---|---|---|---|
| f-LIF, both | spikeDE LIF, both | +0.04 / -0.07 / -0.08 | -0.03 | no clear difference |
| f-LIF, MLP only | spikeDE LIF, MLP only | +0.03 / -0.01 / +0.01 | +0.01 | no clear difference |
| f-LIF, both | ECG model (spikingjelly LIF) | +0.05 / -0.11 / -0.10 | -0.05 | no clear difference |
| f-LIF, MLP only | ECG model (spikingjelly LIF) | +0.05 / -0.03 / -0.07 | -0.02 | no clear difference |
| ECG model (spikingjelly LIF) | no neocortex (ECG ablation) | +0.17 / +0.34 / +0.31 | +0.28 | no clear difference |
| ECG model (spikingjelly LIF) | hippo-only (decision 7) | -0.10 / -0.00 / +0.04 | -0.02 | no clear difference |
| population f-LIF, both | f-LIF, both | +0.00 / +0.06 / +0.05 | +0.03 | no clear difference |
| bidirectional f-LIF | f-LIF, both | -0.02 / +0.10 / -0.04 | +0.01 | no clear difference |
| bidirectional spikeDE LIF | spikeDE LIF, both | +0.04 / +0.11 / +0.00 | +0.05 | no clear difference |
| bidirectional population f-LIF | f-LIF, both | +0.05 / +0.18 / +0.07 | +0.10 | no clear difference |
| bidirectional population f-LIF | bidirectional f-LIF | +0.07 / +0.09 / +0.11 | +0.09 | no clear difference |
| bidirectional population f-LIF | hippo-only (decision 7) | -0.00 / +0.07 / +0.01 | +0.03 | no clear difference |
| bidirectional f-LIF | hippo-only (decision 7) | -0.07 / -0.02 / -0.10 | -0.06 | no clear difference |
| population f-LIF, both | hippo-only (decision 7) | -0.05 / -0.06 / -0.01 | -0.04 | no clear difference |

## A1 replay tests: macro-F1 change when the neocortex output read by the hippocampus is replaced (%p, mean ± sd over seeds)

| condition | normal macro-F1 | another beat | same class | other class | all zero | all one |
|---|---|---|---|---|---|---|
| ECG model (spikingjelly LIF) (n=3) | 91.58 ± 0.06 | -32.95 ± 4.21 | -13.50 ± 1.41 | -38.16 ± 4.45 | -66.62 ± 2.60 | -64.05 ± 4.24 |
| spikeDE LIF, both (n=3) | 91.42 ± 0.29 | -18.55 ± 1.03 | -10.51 ± 2.00 | -20.33 ± 0.52 | -52.17 ± 3.05 | -60.99 ± 2.62 |
| f-LIF, both (n=3) | 91.39 ± 0.42 | -20.05 ± 4.26 | -11.80 ± 2.82 | -21.39 ± 7.72 | -42.75 ± 13.21 | -51.68 ± 3.43 |
| spikeDE LIF, MLP only (n=3) | 91.45 ± 0.23 | -23.69 ± 5.00 | -12.13 ± 1.82 | -29.42 ± 3.72 | -73.21 ± 6.68 | -43.39 ± 7.55 |
| f-LIF, MLP only (n=3) | 91.42 ± 0.20 | -23.15 ± 3.52 | -12.55 ± 2.52 | -26.86 ± 5.38 | -54.60 ± 11.55 | -45.39 ± 19.42 |
| population f-LIF, both (n=3) | 91.32 ± 0.04 | -13.17 ± 4.29 | -8.81 ± 2.66 | -17.13 ± 6.75 | -47.66 ± 5.06 | -48.38 ± 1.69 |
| bidirectional spikeDE LIF (n=3) | 91.65 ± 0.20 | -22.43 ± 5.59 | -10.24 ± 3.31 | -25.71 ± 4.99 | -52.20 ± 21.19 | -48.03 ± 17.14 |
| bidirectional f-LIF (n=3) | 91.23 ± 0.31 | -22.02 ± 3.63 | -12.20 ± 2.28 | -24.70 ± 4.24 | -54.33 ± 39.23 | -45.50 ± 7.85 |
| bidirectional population f-LIF (n=3) | 91.70 ± 0.35 | -15.08 ± 3.12 | -10.93 ± 1.71 | -20.67 ± 3.89 | -50.93 ± 6.41 | -53.22 ± 4.79 |

## A2 linear probes: test macro-F1 of a class-balanced logistic regression (%, mean ± sd over seeds)

| condition | raw beat | neocortex embedding spikes | neocortex output spikes | neocortex output, firing per token | hippocampus head input |
|---|---|---|---|---|---|
| ECG model (spikingjelly LIF) (n=3) | 47.04 ± 0.00 | 61.08 ± 0.83 | 65.53 ± 0.93 | 35.48 ± 1.05 | 87.89 ± 0.43 |
| spikeDE LIF, both (n=3) | 47.04 ± 0.00 | 60.08 ± 0.89 | 60.16 ± 0.21 | 27.84 ± 3.41 | 87.31 ± 0.49 |
| f-LIF, both (n=3) | 47.04 ± 0.00 | 60.80 ± 0.72 | 62.33 ± 0.96 | 28.42 ± 0.56 | 87.11 ± 0.80 |
| spikeDE LIF, MLP only (n=3) | 47.04 ± 0.00 | 61.10 ± 0.35 | 64.01 ± 0.96 | 29.11 ± 1.27 | 88.45 ± 0.40 |
| f-LIF, MLP only (n=3) | 47.04 ± 0.00 | 61.41 ± 0.35 | 64.26 ± 0.24 | 32.15 ± 4.54 | 88.04 ± 0.18 |
| no neocortex (ECG ablation) (n=3) | 47.04 ± 0.00 | — | — | — | 83.49 ± 0.39 |
| hippo-only (decision 7) (n=3) | 47.04 ± 0.00 | — | — | — | 88.25 ± 0.42 |
| population f-LIF, both (n=3) | 47.04 ± 0.00 | 51.73 ± 0.98 | 53.57 ± 0.32 | 28.96 ± 3.79 | 86.97 ± 0.27 |
| bidirectional spikeDE LIF (n=3) | 47.04 ± 0.00 | 58.69 ± 0.95 | 61.72 ± 0.45 | 29.42 ± 2.97 | 88.53 ± 0.58 |
| bidirectional f-LIF (n=3) | 47.04 ± 0.00 | 59.32 ± 0.28 | 63.61 ± 1.39 | 33.58 ± 3.95 | 87.93 ± 0.30 |
| bidirectional population f-LIF (n=3) | 47.04 ± 0.00 | 46.14 ± 3.35 | 47.08 ± 2.13 | 26.41 ± 1.99 | 86.74 ± 0.34 |

## A3 token-axis memory: share of neocortex output spikes that change d tokens after the replaced token (%, mean over seeds)

| condition | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 | d=9 | d=10 | changed predictions, token 0 | max change before the replaced token (expected > 0 for bidirectional) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ECG model (spikingjelly LIF) (n=3) | 15.29 | 12.06 | 9.11 | 6.92 | 5.54 | 4.51 | 3.67 | 2.95 | 2.46 | 2.05 | 1.79 | 5.62 | 0.0e+00 |
| spikeDE LIF, both (n=3) | 9.28 | 9.87 | 6.52 | 3.92 | 2.67 | 1.96 | 1.47 | 1.13 | 0.92 | 0.78 | 0.69 | 2.34 | 0.0e+00 |
| f-LIF, both (n=3) | 10.88 | 12.96 | 4.13 | 3.20 | 2.79 | 2.05 | 1.74 | 1.41 | 1.26 | 1.16 | 1.11 | 2.47 | 0.0e+00 |
| spikeDE LIF, MLP only (n=3) | 10.86 | 9.76 | 6.17 | 3.64 | 2.45 | 1.79 | 1.35 | 1.04 | 0.83 | 0.68 | 0.59 | 3.84 | 0.0e+00 |
| f-LIF, MLP only (n=3) | 12.24 | 11.68 | 4.76 | 3.41 | 2.62 | 1.93 | 1.47 | 1.06 | 0.81 | 0.63 | 0.54 | 3.04 | 0.0e+00 |
| population f-LIF, both (n=3) | 5.46 | 6.65 | 7.78 | 8.22 | 8.30 | 8.04 | 7.60 | 7.21 | 7.06 | 6.93 | 7.07 | 1.76 | 0.0e+00 |
| bidirectional spikeDE LIF (n=3) | 10.14 | 4.97 | 3.57 | 2.37 | 1.73 | 1.34 | 1.07 | 0.88 | 0.74 | 0.63 | 0.55 | 1.68 | 8.3e-02 |
| bidirectional f-LIF (n=3) | 11.99 | 7.44 | 2.54 | 1.91 | 1.70 | 1.25 | 1.08 | 0.85 | 0.75 | 0.66 | 0.63 | 2.43 | 1.4e-01 |
| bidirectional population f-LIF (n=3) | 4.94 | 2.12 | 2.59 | 2.82 | 2.91 | 2.88 | 2.79 | 2.68 | 2.64 | 2.64 | 2.78 | 0.65 | 7.5e-02 |

## A4 neocortex output firing per token, by class (%, mean over seeds; tokens = 32-sample patches, stride 16)

ECG model (spikingjelly LIF) (n=3):

| class | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N | 21.7 | 23.0 | 25.9 | 24.6 | 21.5 | 17.1 | 12.4 | 8.7 | 6.5 | 4.9 | 4.4 |
| S | 22.6 | 24.8 | 23.2 | 22.3 | 21.3 | 18.1 | 14.2 | 11.3 | 8.1 | 7.0 | 6.4 |
| V | 22.5 | 21.6 | 20.8 | 24.8 | 23.2 | 20.5 | 16.2 | 12.5 | 10.0 | 8.0 | 6.0 |
| F | 21.6 | 20.1 | 26.0 | 23.2 | 13.9 | 7.0 | 4.7 | 4.0 | 4.0 | 4.3 | 4.0 |
| Q | 24.4 | 18.1 | 24.6 | 21.2 | 26.7 | 22.9 | 18.6 | 8.0 | 6.2 | 4.1 | 3.8 |

spikeDE LIF, both (n=3):

| class | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N | 12.0 | 21.3 | 25.8 | 26.5 | 24.2 | 19.8 | 14.6 | 10.3 | 7.2 | 5.3 | 4.3 |
| S | 12.0 | 20.5 | 26.4 | 28.4 | 27.7 | 24.6 | 18.4 | 13.6 | 10.0 | 8.0 | 7.2 |
| V | 11.6 | 23.1 | 27.5 | 30.9 | 30.6 | 28.0 | 23.6 | 17.3 | 12.8 | 9.9 | 7.5 |
| F | 11.8 | 19.5 | 23.3 | 22.3 | 14.2 | 8.5 | 4.0 | 3.7 | 3.8 | 3.9 | 3.8 |
| Q | 13.8 | 25.1 | 31.3 | 33.7 | 33.9 | 33.9 | 28.0 | 16.7 | 6.2 | 3.0 | 2.9 |

f-LIF, both (n=3):

| class | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N | 14.5 | 23.1 | 22.7 | 24.0 | 21.9 | 18.9 | 13.6 | 9.9 | 6.9 | 5.1 | 4.1 |
| S | 15.4 | 24.8 | 26.0 | 23.3 | 23.9 | 21.0 | 16.0 | 12.7 | 9.4 | 7.7 | 7.1 |
| V | 15.3 | 26.2 | 25.2 | 29.5 | 26.9 | 25.5 | 20.4 | 15.3 | 11.9 | 9.5 | 6.8 |
| F | 13.4 | 21.6 | 22.0 | 23.4 | 12.5 | 7.3 | 4.1 | 3.8 | 3.6 | 3.6 | 3.5 |
| Q | 15.3 | 26.4 | 29.9 | 30.0 | 30.0 | 32.2 | 23.4 | 13.2 | 4.5 | 4.1 | 3.4 |

population f-LIF, both (n=3):

| class | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N | 10.0 | 17.0 | 19.1 | 21.1 | 21.8 | 21.8 | 20.8 | 19.4 | 17.9 | 16.6 | 15.5 |
| S | 10.1 | 17.2 | 18.9 | 20.8 | 22.4 | 22.2 | 22.3 | 21.4 | 20.3 | 18.9 | 17.7 |
| V | 8.8 | 15.5 | 17.9 | 20.8 | 22.8 | 24.1 | 24.5 | 24.1 | 23.4 | 22.6 | 21.5 |
| F | 9.5 | 17.2 | 20.0 | 20.1 | 19.7 | 17.5 | 14.6 | 13.6 | 12.6 | 11.9 | 11.4 |
| Q | 9.2 | 15.6 | 18.0 | 21.2 | 24.1 | 26.5 | 27.3 | 26.2 | 24.7 | 23.2 | 21.6 |

bidirectional f-LIF (n=3):

| class | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 | t8 | t9 | t10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| N | 18.8 | 23.5 | 22.0 | 23.8 | 21.1 | 19.0 | 14.3 | 10.7 | 7.4 | 6.2 | 5.1 |
| S | 23.9 | 22.1 | 24.2 | 25.1 | 22.5 | 20.9 | 16.0 | 12.8 | 9.6 | 8.6 | 7.2 |
| V | 24.5 | 30.8 | 27.9 | 29.1 | 27.5 | 24.9 | 20.4 | 16.0 | 11.7 | 9.7 | 6.9 |
| F | 16.7 | 26.2 | 18.0 | 20.2 | 14.6 | 11.5 | 5.7 | 5.4 | 4.5 | 5.0 | 4.6 |
| Q | 27.5 | 32.9 | 28.8 | 28.6 | 32.1 | 27.6 | 23.7 | 16.6 | 5.3 | 4.6 | 4.2 |

