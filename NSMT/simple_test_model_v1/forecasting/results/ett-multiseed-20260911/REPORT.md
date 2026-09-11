# ETT multi-seed replication — seeds 7, 13, 21

120 complete results: 40 existing seed-7 runs and 80 new seed-13/21 runs. Unchanged model, training protocol, data hashes and environment versions were verified.

Full ETT splits; input 96; prediction 96/720; patch/stride 8; max 10 epochs; patience 3; best validation MSE checkpoint. All models use train-only scaling and input-window normalization.

K-axis denotes population-token attention; N-axis denotes observed-patch attention. LIF simulation follows N in both variants. N-axis + identity adds a trainable [K,D] population identity; Gaussian centers and widths remain fixed in every population-coded variant. No SSA retains the population frontend, spiking MLP and two-stage head. Linear is the shared-channel normalized linear baseline.

## Seed-level macro statistics

Each seed averages eight dataset/horizon tasks first. Mean ± sample SD (ddof=1) below is over three such seed averages.

| Variant | Macro MSE mean ± SD | Macro MAE mean ± SD |
|---|---:|---:|
| population | 0.38085 ± 0.00160 | 0.40210 ± 0.00125 |
| temporal | 0.38071 ± 0.00070 | 0.40252 ± 0.00060 |
| temporal_embedding | 0.38085 ± 0.00105 | 0.40292 ± 0.00094 |
| no_attention | 0.38171 ± 0.00134 | 0.40297 ± 0.00090 |
| linear | 0.37398 ± 0.00010 | 0.38965 ± 0.00018 |

## Per-task MSE: mean ± sample SD across three seeds

| Dataset | Horizon | K-axis | N-axis | N-axis + identity | No SSA | Linear |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.3927 ± 0.0006 | 0.3902 ± 0.0023 | 0.3894 ± 0.0017 | 0.3943 ± 0.0034 | 0.3877 ± 0.0004 |
| ETTh1 | 720 | 0.4947 ± 0.0091 | 0.4929 ± 0.0084 | 0.4958 ± 0.0039 | 0.4990 ± 0.0099 | 0.4659 ± 0.0009 |
| ETTh2 | 96 | 0.3080 ± 0.0018 | 0.3079 ± 0.0021 | 0.3097 ± 0.0022 | 0.3102 ± 0.0016 | 0.2923 ± 0.0005 |
| ETTh2 | 720 | 0.4372 ± 0.0019 | 0.4352 ± 0.0003 | 0.4359 ± 0.0019 | 0.4374 ± 0.0035 | 0.4209 ± 0.0005 |
| ETTm1 | 96 | 0.3467 ± 0.0015 | 0.3507 ± 0.0006 | 0.3502 ± 0.0028 | 0.3465 ± 0.0021 | 0.3514 ± 0.0022 |
| ETTm1 | 720 | 0.4633 ± 0.0021 | 0.4646 ± 0.0014 | 0.4639 ± 0.0017 | 0.4640 ± 0.0021 | 0.4836 ± 0.0009 |
| ETTm2 | 96 | 0.1838 ± 0.0023 | 0.1836 ± 0.0025 | 0.1813 ± 0.0008 | 0.1820 ± 0.0010 | 0.1823 ± 0.0002 |
| ETTm2 | 720 | 0.4203 ± 0.0080 | 0.4207 ± 0.0086 | 0.4205 ± 0.0115 | 0.4202 ± 0.0086 | 0.4078 ± 0.0006 |

## Per-task MAE: mean ± sample SD across three seeds

| Dataset | Horizon | K-axis | N-axis | N-axis + identity | No SSA | Linear |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.4108 ± 0.0016 | 0.4103 ± 0.0010 | 0.4099 ± 0.0013 | 0.4127 ± 0.0038 | 0.3962 ± 0.0003 |
| ETTh1 | 720 | 0.4830 ± 0.0041 | 0.4831 ± 0.0036 | 0.4838 ± 0.0010 | 0.4847 ± 0.0047 | 0.4592 ± 0.0002 |
| ETTh2 | 96 | 0.3579 ± 0.0018 | 0.3587 ± 0.0027 | 0.3593 ± 0.0025 | 0.3593 ± 0.0018 | 0.3412 ± 0.0006 |
| ETTh2 | 720 | 0.4549 ± 0.0028 | 0.4541 ± 0.0017 | 0.4564 ± 0.0040 | 0.4565 ± 0.0049 | 0.4395 ± 0.0002 |
| ETTm1 | 96 | 0.3808 ± 0.0018 | 0.3823 ± 0.0018 | 0.3832 ± 0.0027 | 0.3805 ± 0.0009 | 0.3719 ± 0.0020 |
| ETTm1 | 720 | 0.4463 ± 0.0013 | 0.4481 ± 0.0015 | 0.4476 ± 0.0017 | 0.4467 ± 0.0015 | 0.4458 ± 0.0002 |
| ETTm2 | 96 | 0.2681 ± 0.0015 | 0.2683 ± 0.0017 | 0.2681 ± 0.0023 | 0.2681 ± 0.0033 | 0.2650 ± 0.0005 |
| ETTm2 | 720 | 0.4150 ± 0.0045 | 0.4152 ± 0.0043 | 0.4151 ± 0.0062 | 0.4153 ± 0.0051 | 0.3983 ± 0.0005 |

## Paired MSE comparisons

Deltas are tested minus reference, matched by task and seed; negative means improvement. Macro delta SD is over three seed-level means, not over 24 heterogeneous task/seed pairs.

| Tested vs reference | Macro ΔMSE mean ± SD | Wins / 24 seed-task pairs | Tasks won on mean / 8 | Tasks won in all seeds / 8 |
|---|---:|---:|---:|---:|
| temporal vs population | -0.00013 ± 0.00115 | 12/24 | 5/8 | 1/8 |
| temporal_embedding vs temporal | 0.00013 ± 0.00077 | 14/24 | 5/8 | 2/8 |
| temporal vs no_attention | -0.00099 ± 0.00128 | 10/24 | 4/8 | 0/8 |
| temporal vs linear | 0.00673 ± 0.00067 | 6/24 | 2/8 | 1/8 |

## Limits

Three seeds describe initialization/training-order variability under this short fixed protocol. They do not establish convergence or statistical significance. All seeds reuse the same data splits; datasets/horizons are not independent replications of initialization. The seed-7 result motivated this replication; model/settings were frozen before the two new seeds. Pairing matches the random seed and task, but does not guarantee identical shared-layer initialization across architectures with different parameter sets (including No SSA). No-population SNN, flatten-head, longer-budget and separately tuned attention-scale comparisons were not run. See per_run.csv, per_task.csv, paired.csv and provenance.json for all inputs and paired results.
