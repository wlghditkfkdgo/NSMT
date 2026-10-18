# MIT-BIH suite `flif-v1-cls` — summary

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

## Pre-registered comparisons (macro-F1, paired by seed, %p)

| question | A | B | A − B per seed (42 / 7 / 2026) | mean | verdict |
|---|---|---|---|---|---|
| Q1 f-LIF vs spikeDE LIF | f-LIF, both | spikeDE LIF, both | +0.19 / -0.04 / -0.22 | -0.02 | no clear difference |
| Q1 f-LIF vs spikeDE LIF | f-LIF, MLP only | spikeDE LIF, MLP only | -0.08 / +0.25 / -0.26 | -0.03 | no clear difference |
| Q2 f-LIF vs ECG model | f-LIF, both | ECG model (spikingjelly LIF) | +0.22 / -0.51 / -0.29 | -0.19 | no clear difference |
| Q2 f-LIF vs ECG model | f-LIF, MLP only | ECG model (spikingjelly LIF) | -0.05 / -0.10 / -0.33 | -0.16 | no clear difference |
| Q3 neocortex contribution | ECG model (spikingjelly LIF) | no neocortex (ECG ablation) | +0.86 / +1.29 / +1.40 | +1.18 | better |
| Q3 neocortex contribution | ECG model (spikingjelly LIF) | hippo-only (decision 7) | -0.42 / -0.25 / -0.07 | -0.25 | no clear difference |

Accuracy, same rule:

| A | B | A − B per seed | mean | verdict |
|---|---|---|---|---|
| f-LIF, both | spikeDE LIF, both | +0.04 / -0.07 / -0.08 | -0.03 | no clear difference |
| f-LIF, MLP only | spikeDE LIF, MLP only | +0.03 / -0.01 / +0.01 | +0.01 | no clear difference |
| f-LIF, both | ECG model (spikingjelly LIF) | +0.05 / -0.11 / -0.10 | -0.05 | no clear difference |
| f-LIF, MLP only | ECG model (spikingjelly LIF) | +0.05 / -0.03 / -0.07 | -0.02 | no clear difference |
| ECG model (spikingjelly LIF) | no neocortex (ECG ablation) | +0.17 / +0.34 / +0.31 | +0.28 | no clear difference |
| ECG model (spikingjelly LIF) | hippo-only (decision 7) | -0.10 / -0.00 / +0.04 | -0.02 | no clear difference |

## A1 replay tests: macro-F1 change when the neocortex output read by the hippocampus is replaced (%p, mean ± sd over seeds)

| condition | normal macro-F1 | another beat | same class | other class | all zero | all one |
|---|---|---|---|---|---|---|
| ECG model (spikingjelly LIF) (n=3) | 91.58 ± 0.06 | -32.95 ± 4.21 | -13.50 ± 1.41 | -38.16 ± 4.45 | -66.62 ± 2.60 | -64.05 ± 4.24 |
| spikeDE LIF, both (n=3) | 91.42 ± 0.29 | -18.55 ± 1.03 | -10.51 ± 2.00 | -20.33 ± 0.52 | -52.17 ± 3.05 | -60.99 ± 2.62 |
| f-LIF, both (n=3) | 91.39 ± 0.42 | -20.05 ± 4.26 | -11.80 ± 2.82 | -21.39 ± 7.72 | -42.75 ± 13.21 | -51.68 ± 3.43 |
| spikeDE LIF, MLP only (n=3) | 91.45 ± 0.23 | -23.69 ± 5.00 | -12.13 ± 1.82 | -29.42 ± 3.72 | -73.21 ± 6.68 | -43.39 ± 7.55 |
| f-LIF, MLP only (n=3) | 91.42 ± 0.20 | -23.15 ± 3.52 | -12.55 ± 2.52 | -26.86 ± 5.38 | -54.60 ± 11.55 | -45.39 ± 19.42 |

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

## A3 token-axis memory: share of neocortex output spikes that change d tokens after the replaced token (%, mean over seeds)

| condition | d=0 | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 | d=7 | d=8 | d=9 | d=10 | changed predictions, token 0 | max change before the replaced token |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ECG model (spikingjelly LIF) (n=3) | 15.29 | 12.06 | 9.11 | 6.92 | 5.54 | 4.51 | 3.67 | 2.95 | 2.46 | 2.05 | 1.79 | 5.62 | 0.0e+00 |
| spikeDE LIF, both (n=3) | 9.28 | 9.87 | 6.52 | 3.92 | 2.67 | 1.96 | 1.47 | 1.13 | 0.92 | 0.78 | 0.69 | 2.34 | 0.0e+00 |
| f-LIF, both (n=3) | 10.88 | 12.96 | 4.13 | 3.20 | 2.79 | 2.05 | 1.74 | 1.41 | 1.26 | 1.16 | 1.11 | 2.47 | 0.0e+00 |
| spikeDE LIF, MLP only (n=3) | 10.86 | 9.76 | 6.17 | 3.64 | 2.45 | 1.79 | 1.35 | 1.04 | 0.83 | 0.68 | 0.59 | 3.84 | 0.0e+00 |
| f-LIF, MLP only (n=3) | 12.24 | 11.68 | 4.76 | 3.41 | 2.62 | 1.93 | 1.47 | 1.06 | 0.81 | 0.63 | 0.54 | 3.04 | 0.0e+00 |

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

