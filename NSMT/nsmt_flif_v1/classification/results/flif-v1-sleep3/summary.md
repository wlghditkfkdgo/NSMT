# Sleep-EDF: Neocortex membrane readout and past-reconstruction aux (lr 0.003)

Pre-registered: docs/MODI_PROJECT_LOG.md 2026-11-27. Code versions: [{'ff583b0bc'}, {'ff583b0bc'}, {'43afbb75e'}, {'3db4bebde'}]. Test macro-F1 %, folds 0-4, paired by fold.

| model | macro-F1 | accuracy | N1 F1 | syops energy µJ | × Spikformer(DE) |
|---|---|---|---|---|---|
| spkall_lif | 74.03 ± 3.27 | 81.06 | 34.8 | 16.12 | 1.00 |
| q_pop_lif | 74.16 ± 3.26 | 80.50 | 37.4 | 18.55 | 1.15 |
| q_pop_lif_cshuf | 73.63 ± 3.15 | 80.40 | 36.2 | 18.54 | 1.15 |
| q_pop_lif_nml | 73.41 ± 2.74 | 80.17 | 36.0 | 16.23 | 1.01 |
| q_pop_lif_mem | 74.63 ± 3.40 | 81.51 | 36.1 | 27.84 | 1.73 |
| q_pop_lif_acc | 73.87 ± 3.79 | 80.51 | 35.8 | 19.29 | 1.20 |
| q_pop_lif_mem_acc | 74.17 ± 3.72 | 81.00 | 35.2 | 27.88 | 1.73 |
| q_pop_lif_mem_cshuf | 74.11 ± 2.80 | 81.05 | 36.0 | 27.88 | 1.73 |
| q_pop_lif_mem_nml | 74.60 ± 2.66 | 81.33 | 36.7 | 27.08 | 1.68 |
| q_pop_lif_acc_cshuf | 73.76 ± 3.17 | 80.84 | 34.0 | 19.48 | 1.21 |
| q_pop_flif | 73.35 ± 3.29 | 79.96 | 36.5 | 18.63 | 1.16 |
| q_pop_flif_cshuf | 73.99 ± 3.19 | 80.74 | 37.0 | 18.45 | 1.14 |
| q_pop_flif_nml | 72.89 ± 4.06 | 79.96 | 34.6 | 16.49 | 1.02 |
| q_pop_flif_mem | 73.23 ± 3.55 | 79.89 | 36.6 | 27.80 | 1.72 |
| q_pop_flif_acc | 73.94 ± 2.26 | 80.77 | 37.1 | 19.44 | 1.21 |
| q_pop_flif_mem_acc | 74.24 ± 3.57 | 80.76 | 37.0 | 27.76 | 1.72 |
| q_pop_flif_mem_cshuf | 74.54 ± 3.14 | 81.24 | 37.2 | 27.70 | 1.72 |
| q_pop_flif_mem_nml | 73.36 ± 2.44 | 80.50 | 33.9 | 27.24 | 1.69 |
| q_pop_flif_acc_cshuf | 74.21 ± 3.21 | 80.94 | 36.6 | 19.66 | 1.22 |

| test | neuron | d per fold | mean | positive | verdict |
|---|---|---|---|---|---|
| E1 mem − spike | pop_lif | +0.36 / +2.79 / -1.07 / +0.21 / +0.09 | +0.48 | 4/5 | **no clear difference** |
| E1 mem − spike | pop_flif | +0.94 / -0.02 / -0.06 / -1.05 / -0.42 | -0.12 | 1/5 | **no clear difference** |
| E2 acc − spike | pop_lif | -1.47 / +4.27 / -0.47 / -1.60 / -2.15 | -0.28 | 1/5 | **no clear difference** |
| E2 acc − spike | pop_flif | -0.85 / +1.25 / -0.13 / +0.11 / +2.52 | +0.58 | 3/5 | **no clear difference** |
| E3 mem+acc − spike | pop_lif | -0.16 / +0.84 / +0.88 / +0.04 / -1.53 | +0.01 | 3/5 | **no clear difference** |
| E3 mem+acc − spike | pop_flif | +0.93 / +2.83 / +0.54 / -0.18 / +0.30 | +0.88 | 4/5 | **no clear difference** |
| S mem − Spikformer(DE) | pop_lif | +0.62 / -0.56 / -0.02 / +2.42 / +0.56 | +0.60 | 3/5 | **no clear difference** |
| S mem − Spikformer(DE) | pop_flif | -0.49 / -2.50 / +1.01 / -0.24 / -1.76 | -0.80 | 1/5 | **no clear difference** |
| S acc − Spikformer(DE) | pop_lif | -1.21 / +0.93 / +0.57 / +0.61 / -1.69 | -0.16 | 3/5 | **no clear difference** |
| S acc − Spikformer(DE) | pop_flif | -2.27 / -1.23 / +0.94 / +0.92 / +1.17 | -0.09 | 3/5 | **no clear difference** |
| S mem+acc − Spikformer(DE) | pop_lif | +0.10 / -2.50 / +1.92 / +2.25 / -1.06 | +0.14 | 3/5 | **no clear difference** |
| S mem+acc − Spikformer(DE) | pop_flif | -0.49 / +0.35 / +1.61 / +0.62 / -1.04 | +0.21 | 3/5 | **no clear difference** |
| O mem − mem_cshuf | pop_lif | +0.84 / +1.09 / -0.68 / +1.62 / -0.24 | +0.52 | 3/5 | **no clear difference** |
| O mem − mem_cshuf | pop_flif | -0.58 / -3.31 / +1.06 / -1.04 / -2.68 | -1.31 | 1/5 | **worse** |
| O acc − acc_cshuf | pop_lif | -0.67 / +1.88 / +0.99 / -0.09 / -1.53 | +0.12 | 2/5 | **no clear difference** |
| O acc − acc_cshuf | pop_flif | -1.44 / -2.81 / +0.74 / +1.38 / +0.77 | -0.27 | 3/5 | **no clear difference** |
| M mem − mem_nml | pop_lif | +1.32 / -0.86 / -0.38 / +1.14 / -1.04 | +0.03 | 2/5 | **no clear difference** |
| M mem − mem_nml | pop_flif | +1.47 / -0.04 / +0.59 / -0.48 / -2.18 | -0.13 | 2/5 | **no clear difference** |

**Reading (pre-registered):** mem / pop_lif: neither; acc / pop_lif: neither; mem_acc / pop_lif: does not beat Spikformer(DE); mem / pop_flif: neither; acc / pop_flif: neither; mem_acc / pop_flif: does not beat Spikformer(DE)
