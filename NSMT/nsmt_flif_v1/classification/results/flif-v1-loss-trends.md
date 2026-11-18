# Loss trends of the current experiments (per-epoch logs; no training) — summary

See the module docstring for the definitions. Values are medians over the runs of a row unless a count. Train / validation L1 (ETT) are on different splits and modes; the gap is descriptive.

## ETT by model and input length (12 runs per row: 4 cells x 3 seeds)

| input | model | epochs (median, min–max) | best epoch (median; # at epoch 0) | train change, epoch 0 → end % | train change after best % | train fell in % of epochs | train, last 3 epochs % | val change, epoch 0 → best % | val change, best → end % | gap val − train at best / end | # overfit | # train still falling | # best at last epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| input 96 | Spikformer-style | 14 (11–24) | 3; 2 | -8.1 | -3.7 | 86 | -0.17 | -1.0 | +1.2 | -0.013 / -0.009 | 7 | 0 | 0 |
| input 96 | base pop LIF | 13 (11–22) | 2; 1 | -7.7 | -4.0 | 86 | -0.17 | -0.3 | +1.0 | -0.022 / -0.006 | 6 | 1 | 0 |
| input 96 | base pop f-LIF | 12 (11–20) | 2; 2 | -7.8 | -3.6 | 87 | -0.16 | -0.5 | +1.0 | -0.020 / -0.009 | 6 | 1 | 0 |
| input 96 SpikeDE | Spikformer-style SpikeDE | 14 (11–16) | 2; 2 | -9.9 | -5.3 | 92 | -0.26 | -0.6 | +1.8 | -0.017 / -0.000 | 8 | 1 | 0 |
| input 96 SpikeDE | base pop LIF SpikeDE | 12 (11–18) | 2; 3 | -9.1 | -5.7 | 90 | -0.26 | -0.3 | +1.5 | -0.023 / -0.005 | 8 | 1 | 0 |
| input 96 SpikeDE | base pop f-LIF SpikeDE | 12 (11–18) | 2; 4 | -9.2 | -5.4 | 88 | -0.26 | -0.3 | +1.4 | -0.020 / -0.005 | 8 | 2 | 0 |
| input 96 XNOR | Spikformer-style XNOR | 12 (11–18) | 2; 3 | -8.0 | -3.9 | 90 | -0.15 | -0.8 | +1.1 | -0.018 / -0.006 | 6 | 0 | 0 |
| input 96 XNOR | base pop LIF XNOR | 12 (11–18) | 2; 3 | -8.5 | -4.9 | 90 | -0.18 | -0.5 | +1.3 | -0.018 / -0.004 | 7 | 0 | 0 |
| input 96 XNOR | base pop f-LIF XNOR | 12 (11–37) | 1; 2 | -8.8 | -4.8 | 83 | -0.19 | -0.7 | +1.2 | -0.021 / -0.006 | 7 | 1 | 0 |
| input 336 | Spikformer-style | 12 (11–15) | 0; 6 | -18.3 | -13.8 | 92 | -0.44 | -0.0 | +1.7 | -0.001 / +0.041 | 11 | 5 | 0 |
| input 336 | base pop LIF | 11 (11–14) | 0; 7 | -19.0 | -15.8 | 96 | -0.52 | +0.0 | +2.1 | -0.013 / +0.046 | 11 | 7 | 0 |
| input 336 | base pop f-LIF | 11 (11–13) | 0; 9 | -18.8 | -16.0 | 91 | -0.46 | +0.0 | +2.1 | -0.020 / +0.039 | 12 | 5 | 0 |
| input 720 | Spikformer-style | 11 (11–15) | 0; 9 | -26.0 | -24.1 | 100 | -0.74 | +0.0 | +4.2 | -0.001 / +0.097 | 12 | 12 | 0 |
| input 720 | base pop LIF | 11 (11–13) | 0; 9 | -26.6 | -25.2 | 100 | -0.77 | +0.0 | +3.7 | +0.003 / +0.097 | 12 | 12 | 0 |
| input 720 | base pop f-LIF | 11 (11–12) | 0; 10 | -27.0 | -25.3 | 100 | -0.77 | +0.0 | +3.1 | +0.001 / +0.087 | 12 | 12 | 0 |

## ETT by cell (input 96 base and Spikformer-style, spikingjelly; 3 seeds per row)

| cell | model | best epoch per seed | epochs per seed | train change after best % (per seed) | val change best → end % (per seed) | overfit |
|---|---|---|---|---|---|---|
| ETTh2 H96 | Spikformer-style | 7 / 3 / 1 | 18 / 14 / 12 | -0.5 / -0.5 / -5.9 | +0.4 / +1.2 / +1.2 | 1/3 |
| ETTh2 H96 | base pop LIF | 1 / 1 / 1 | 12 / 12 / 12 | -2.9 / -5.4 / -5.4 | +0.5 / +0.9 / +1.6 | 1/3 |
| ETTh2 H96 | base pop f-LIF | 1 / 1 / 1 | 12 / 12 / 12 | -3.3 / -5.6 / -5.6 | +0.5 / +0.9 / +1.2 | 1/3 |
| ETTh2 H720 | Spikformer-style | 4 / 3 / 3 | 15 / 14 / 14 | -4.2 / -3.7 / -3.7 | +2.7 / +2.0 / +1.8 | 3/3 |
| ETTh2 H720 | base pop LIF | 4 / 4 / 3 | 15 / 15 / 14 | -2.5 / -4.1 / -4.0 | +1.2 / +2.0 / +2.1 | 3/3 |
| ETTh2 H720 | base pop f-LIF | 2 / 4 / 3 | 13 / 15 / 14 | -3.6 / -3.6 / -3.5 | +0.9 / +2.3 / +1.2 | 2/3 |
| ETTm2 H96 | Spikformer-style | 13 / 11 / 7 | 24 / 22 / 18 | +0.2 / -0.3 / -1.0 | +0.0 / +0.2 / +0.1 | 0/3 |
| ETTm2 H96 | base pop LIF | 5 / 11 / 4 | 16 / 22 / 15 | -3.3 / -0.4 / -3.1 | +0.1 / +0.2 / +0.1 | 0/3 |
| ETTm2 H96 | base pop f-LIF | 7 / 6 / 9 | 18 / 17 / 20 | -2.6 / +0.2 / -2.5 | +0.7 / +0.1 / +0.5 | 0/3 |
| ETTm2 H720 | Spikformer-style | 1 / 0 / 0 | 12 / 11 / 11 | -4.1 / -5.4 / -5.5 | +1.2 / +1.4 / +1.0 | 3/3 |
| ETTm2 H720 | base pop LIF | 0 / 1 / 1 | 11 / 12 / 12 | -5.2 / -4.0 / -4.4 | +1.0 / +1.1 / +1.2 | 2/3 |
| ETTm2 H720 | base pop f-LIF | 0 / 0 / 1 | 11 / 11 / 12 | -5.5 / -5.6 / -4.4 | +1.0 / +1.4 / +1.6 | 3/3 |

## MIT-BIH (30 epochs, cosine lr, last epoch tested; "test" = the per-epoch test-file columns, read only)

| model | train CE epoch 0 → end % | train CE last 5 epochs % | test CE min epoch (per seed) | test CE rise min → end % (per seed) | test F1 max epoch (per seed) | F1 max → end %p (per seed) | train / test CE at the end | # overfit |
|---|---|---|---|---|---|---|---|---|
| Spikformer-style | -99.5 | -38.4 | 28 / 29 / 28 | +0.8 / +0.0 / +0.4 | 26 / 27 / 27 | -0.26 / -0.04 / -0.17 | 0.0074 / 0.0561 | 0/3 |
| base pop LIF | -99.5 | -39.1 | 26 / 27 / 25 | +1.2 / +1.1 / +3.5 | 28 / 26 / 25 | -0.07 / -0.70 / -0.40 | 0.0072 / 0.0587 | 0/3 |
| base pop f-LIF | -99.5 | -40.2 | 29 / 25 / 28 | +0.0 / +0.5 / +1.1 | 29 / 25 / 28 | +0.00 / -0.13 / -0.24 | 0.0075 / 0.0570 | 0/3 |
| Spikformer-style SpikeDE | -99.6 | -41.0 | 20 / 27 / 25 | +2.3 / +0.9 / +0.2 | 25 / 24 / 27 | -0.21 / -0.45 / -0.12 | 0.0069 / 0.0576 | 0/3 |
| base pop LIF SpikeDE | -99.6 | -43.4 | 24 / 22 / 25 | +2.9 / +7.1 / +2.5 | 25 / 27 / 27 | -0.04 / -0.23 / -0.06 | 0.0060 / 0.0626 | 0/3 |
| base pop f-LIF SpikeDE | -99.6 | -46.1 | 24 / 23 / 25 | +0.9 / +2.2 / +1.5 | 27 / 27 / 28 | -0.11 / -0.34 / -0.04 | 0.0059 / 0.0594 | 0/3 |
