# ETT quick validation — ett-quick-20260910

Completed 40/40 predefined runs.

Seed 7; full canonical ETT splits; input 96; horizon 96/720; patch/stride 8; maximum 10 epochs, patience 3; best validation MSE checkpoint. Metrics are on the train-standardized scale, averaged over all windows/horizon points/channels.

All SNN variants use fixed Gaussian coding, direct input currents and the same two-stage head. Temporal embedding adds a trainable population identity. No-attention keeps embedding and IAND MLP. Linear is a shared per-channel Linear(96,H) with the same input-window normalization.

| Dataset | Horizon | K-axis MSE/MAE | N-axis MSE/MAE | N-axis + identity MSE/MAE | No SSA MSE/MAE | Linear MSE/MAE |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.3920 / 0.4090 | 0.3915 / 0.4091 | 0.3900 / 0.4084 | 0.3939 / 0.4104 | 0.3874 / 0.3962 |
| ETTh1 | 720 | 0.5038 / 0.4866 | 0.5025 / 0.4872 | 0.4999 / 0.4850 | 0.5020 / 0.4859 | 0.4648 / 0.4590 |
| ETTh2 | 96 | 0.3080 / 0.3572 | 0.3097 / 0.3616 | 0.3112 / 0.3618 | 0.3094 / 0.3579 | 0.2920 / 0.3408 |
| ETTh2 | 720 | 0.4375 / 0.4543 | 0.4352 / 0.4534 | 0.4380 / 0.4603 | 0.4405 / 0.4610 | 0.4204 / 0.4395 |
| ETTm1 | 96 | 0.3455 / 0.3807 | 0.3502 / 0.3844 | 0.3533 / 0.3861 | 0.3464 / 0.3815 | 0.3539 / 0.3741 |
| ETTm1 | 720 | 0.4637 / 0.4462 | 0.4642 / 0.4475 | 0.4643 / 0.4473 | 0.4634 / 0.4464 | 0.4835 / 0.4458 |
| ETTm2 | 96 | 0.1862 / 0.2695 | 0.1864 / 0.2702 | 0.1816 / 0.2708 | 0.1831 / 0.2718 | 0.1820 / 0.2646 |
| ETTm2 | 720 | 0.4136 / 0.4119 | 0.4110 / 0.4103 | 0.4095 / 0.4087 | 0.4113 / 0.4104 | 0.4085 / 0.3988 |

Macro averages weight the eight dataset/horizon tasks equally; they are not pooled errors.

| Variant | Tasks | Macro MSE | Macro MAE |
|---|---:|---:|---:|
| population | 8 | 0.3813 | 0.4019 |
| temporal | 8 | 0.3814 | 0.4030 |
| temporal_embedding | 8 | 0.3810 | 0.4036 |
| no_attention | 8 | 0.3813 | 0.4032 |
| linear | 8 | 0.3741 | 0.3899 |

## Paired comparisons

Negative relative MSE means the tested variant improved over the reference.

| Tested vs reference | MSE wins/tasks | Mean relative MSE |
|---|---:|---:|
| temporal vs linear | 2/8 | +2.09% |
| temporal vs no_attention | 3/8 | +0.18% |
| temporal vs population | 4/8 | +0.08% |
| temporal_embedding vs temporal | 4/8 | -0.23% |

## Limits

One seed and a short epoch budget provide screening evidence, not statistical significance or convergence. No hyperparameters were selected using test scores. The two-stage head is shared by all SNN conditions; this matrix does not separately establish its advantage over the old flatten head. No-population SNN and multiple-seed comparisons were not run. IAND only suppresses spikes; population and temporal axes use the same fixed scale=1. Their individually optimal scales may differ. Spike statistics cover the first evaluation batch only.

See summary.csv for epochs, validation scores, checkpoints and parameters; comparisons.csv for paired differences; each run JSON contains the exact command, code/data hashes, environment and epoch history.
