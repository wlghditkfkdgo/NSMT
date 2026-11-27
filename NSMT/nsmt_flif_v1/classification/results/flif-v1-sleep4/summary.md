# Sleep-EDF: plain LIF / f-LIF Neocortex (structure, order / memory, membrane readout, accumulation aux)

Pre-registered: docs/MODI_PROJECT_LOG.md 2026-11-27. Code versions: {'flif-v1-sleep-lr': ['ff583b0bc'], 'flif-v1-sleep': ['ff583b0bc'], 'flif-v1-sleep2': ['43afbb75e'], 'flif-v1-sleep3': ['3db4bebde'], 'flif-v1-sleep4': ['091d93529']}. Test macro-F1 %, folds 0-4, paired by fold.

## Three default neurons side by side (lr 0.003; — = not run)

| model | LIF | f-LIF | pop LIF |
|---|---|---|---|
| Spikformer(DE) | 74.03 ± 3.27 (no Neocortex) | | |
| query, spike | 73.39 ± 4.44 | 74.40 ± 3.66 | 74.16 ± 3.26 |
| key/value, spike | 73.70 ± 4.04 | — | 72.33 ± 4.90 |
| query, membrane | 73.45 ± 4.19 | 74.04 ± 4.05 | 74.63 ± 3.40 |
| query, accumulation aux | 75.41 ± 1.98 | — | 73.87 ± 3.79 |
| query, both | 74.65 ± 3.19 | — | 74.17 ± 3.72 |
| query, context shuffled | 74.61 ± 2.83 | — | 73.63 ± 3.15 |
| query, memoryless Neocortex | 74.80 ± 3.84 | — | 73.41 ± 2.74 |
| membrane + context shuffled | 73.39 ± 2.69 | — | 74.11 ± 2.80 |
| membrane + memoryless | 73.77 ± 2.75 | — | 74.60 ± 2.66 |
| aux + context shuffled | 73.16 ± 3.71 | — | 73.76 ± 3.17 |

## Every run set

| model | lr | macro-F1 | accuracy | N1 F1 | syops energy µJ | × Spikformer(DE) @0.003 | firing rate |
|---|---|---|---|---|---|---|---|
| kv_lif | 0.001 | 73.48 ± 3.43 | 80.35 | 35.3 | 18.80 | 1.17 | 0.4176 |
| kv_lif | 0.003 | 73.70 ± 4.04 | 80.34 | 36.6 | 18.78 | 1.16 | 0.4206 |
| kv_pop_lif | 0.003 | 72.33 ± 4.90 | 78.81 | 35.5 | 16.80 | 1.04 | 0.4460 |
| q_flif | 0.003 | 74.40 ± 3.66 | 81.29 | 36.4 | 19.22 | 1.19 | 0.4262 |
| q_flif_mem | 0.003 | 74.04 ± 4.05 | 80.61 | 37.2 | 27.77 | 1.72 | 0.4502 |
| q_lif | 0.001 | 74.00 ± 3.50 | 80.82 | 36.9 | 19.32 | 1.20 | 0.4231 |
| q_lif | 0.003 | 73.39 ± 4.44 | 80.42 | 33.9 | 19.53 | 1.21 | 0.4287 |
| q_lif_acc | 0.003 | 75.41 ± 1.98 | 81.89 | 39.8 | 19.70 | 1.22 | 0.3946 |
| q_lif_acc_cshuf | 0.003 | 73.16 ± 3.71 | 80.83 | 31.8 | 19.51 | 1.21 | 0.3922 |
| q_lif_cshuf | 0.003 | 74.61 ± 2.83 | 81.34 | 36.2 | 19.32 | 1.20 | 0.4261 |
| q_lif_mem | 0.003 | 73.45 ± 4.19 | 79.94 | 36.5 | 27.92 | 1.73 | 0.4541 |
| q_lif_mem_acc | 0.003 | 74.65 ± 3.19 | 81.30 | 39.2 | 27.88 | 1.73 | 0.4144 |
| q_lif_mem_cshuf | 0.003 | 73.39 ± 2.69 | 80.21 | 35.1 | 27.97 | 1.74 | 0.4542 |
| q_lif_mem_nml | 0.003 | 73.77 ± 2.75 | 80.43 | 35.8 | 27.40 | 1.70 | 0.4399 |
| q_lif_nml | 0.003 | 74.80 ± 3.84 | 81.35 | 37.1 | 18.11 | 1.12 | 0.4186 |
| q_pop_lif | 0.001 | 74.62 ± 3.88 | 81.06 | 36.9 | 18.19 | 1.13 | 0.4492 |
| q_pop_lif | 0.003 | 74.16 ± 3.26 | 80.50 | 37.4 | 18.55 | 1.15 | 0.4531 |
| q_pop_lif_acc | 0.003 | 73.87 ± 3.79 | 80.51 | 35.8 | 19.29 | 1.20 | 0.4151 |
| q_pop_lif_acc_cshuf | 0.003 | 73.76 ± 3.17 | 80.84 | 34.0 | 19.48 | 1.21 | 0.4154 |
| q_pop_lif_cshuf | 0.003 | 73.63 ± 3.15 | 80.40 | 36.2 | 18.54 | 1.15 | 0.4533 |
| q_pop_lif_mem | 0.003 | 74.63 ± 3.40 | 81.51 | 36.1 | 27.84 | 1.73 | 0.4801 |
| q_pop_lif_mem_acc | 0.003 | 74.17 ± 3.72 | 81.00 | 35.2 | 27.88 | 1.73 | 0.4369 |
| q_pop_lif_mem_cshuf | 0.003 | 74.11 ± 2.80 | 81.05 | 36.0 | 27.88 | 1.73 | 0.4798 |
| q_pop_lif_mem_nml | 0.003 | 74.60 ± 2.66 | 81.33 | 36.7 | 27.08 | 1.68 | 0.4520 |
| q_pop_lif_nml | 0.003 | 73.41 ± 2.74 | 80.17 | 36.0 | 16.23 | 1.01 | 0.4214 |
| spkall_lif | 0.001 | 73.06 ± 3.65 | 79.75 | 36.6 | 16.14 | 1.00 | 0.4039 |
| spkall_lif | 0.003 | 74.03 ± 3.27 | 81.06 | 34.8 | 16.12 | 1.00 | 0.4055 |

lr choice (fold-0 validation, tie -> 0.001): q_lif 0.7638 / 0.7752 -> 0.003; kv_lif 0.7785 / 0.7874 -> 0.003

| test | d per fold | mean | positive | verdict |
|---|---|---|---|---|
| Q1 q_lif − Spikformer(DE) (chosen lrs) | +1.35 / +1.92 / -0.32 / -0.31 / -0.98 | +0.33 | 2/5 | **no clear difference** |
| Q1 kv_lif − Spikformer(DE) (chosen lrs) | +0.70 / +1.34 / +1.55 / -0.00 / -0.39 | +0.64 | 3/5 | **no clear difference** |
| Q2 q_lif − kv_lif (chosen lrs) | +0.65 / +0.58 / -1.87 / -0.31 / -0.59 | -0.31 | 2/5 | **no clear difference** |
| L q_lif − Spikformer(DE) @0.003 | -0.07 / -0.13 / -0.51 / +0.70 / -3.17 | -0.64 | 1/5 | **no clear difference** |
| L q_lif − Spikformer(DE) @0.001 | +0.97 / +0.83 / +0.60 / +0.90 / +1.36 | +0.93 | 5/5 | **no clear difference** |
| Q2 q_lif − kv_lif @0.003 (report) | +0.65 / +0.58 / -1.87 / -0.31 / -0.59 | -0.31 | 2/5 | no clear difference |
| kv_lif − Spikformer(DE) @0.003 (report) | -0.72 / -0.71 / +1.36 / +1.01 / -2.59 | -0.33 | 2/5 | no clear difference |
| Q2 q_lif − kv_lif @0.001 (report) | +0.88 / +0.35 / -0.15 / +0.86 / +0.66 | +0.52 | 4/5 | no clear difference |
| kv_lif − Spikformer(DE) @0.001 (report) | +0.10 / +0.48 / +0.75 / +0.04 / +0.70 | +0.41 | 5/5 | no clear difference |
| O q_lif − q_lif_cshuf | +1.88 / -2.09 / -1.40 / -0.03 / -4.46 | -1.22 | 1/5 | **worse** |
| M q_lif − q_lif_nml | +1.37 / -2.79 / -3.27 / -0.10 / -2.24 | -1.41 | 1/5 | **worse** |
| E q_lif_mem − q_lif | -0.27 / -2.19 / +1.96 / +0.57 / +0.21 | +0.06 | 3/5 | **no clear difference** |
| E q_lif_acc − q_lif | -0.11 / -0.10 / +2.97 / +0.80 / +6.53 | +2.02 | 3/5 | **no clear difference** |
| E q_lif_mem_acc − q_lif | +0.52 / +0.16 / -0.12 / +1.28 / +4.46 | +1.26 | 4/5 | **better** |
| E q_flif_mem − q_flif | -0.54 / -1.09 / +1.27 / +0.26 / -1.68 | -0.35 | 2/5 | **no clear difference** |
| S q_lif_mem − Spikformer(DE) @0.003 | -0.34 / -2.32 / +1.45 / +1.26 / -2.96 | -0.58 | 2/5 | **no clear difference** |
| S q_lif_acc − Spikformer(DE) @0.003 | -0.18 / -0.23 / +2.46 / +1.50 / +3.35 | +1.38 | 3/5 | **no clear difference** |
| S q_lif_mem_acc − Spikformer(DE) @0.003 | +0.44 / +0.04 / -0.63 / +1.98 / +1.29 | +0.62 | 4/5 | **no clear difference** |
| S q_flif − Spikformer(DE) @0.003 | +0.64 / -0.25 / -0.18 / +1.92 / -0.30 | +0.37 | 2/5 | **no clear difference** |
| S q_flif_mem − Spikformer(DE) @0.003 | +0.10 / -1.34 / +1.09 / +2.19 / -1.98 | +0.01 | 3/5 | **no clear difference** |
| O q_lif_mem − q_lif_mem_cshuf | +2.09 / +0.82 / -0.14 / +0.38 / -2.83 | +0.06 | 3/5 | **no clear difference** |
| O q_lif_acc − q_lif_acc_cshuf | -0.20 / +3.65 / +3.16 / +0.04 / +4.57 | +2.25 | 4/5 | **better** |
| M q_lif_mem − q_lif_mem_nml | +0.97 / -0.92 / +1.46 / +0.28 / -3.38 | -0.32 | 3/5 | **no clear difference** |
| N q_lif − q_pop_lif @0.003 | -0.34 / +3.22 / -1.55 / -1.52 / -3.64 | -0.77 | 1/5 | **no clear difference** |
| N q_flif − q_pop_lif @0.003 | +0.38 / +3.09 / -1.22 / -0.29 / -0.77 | +0.24 | 2/5 | **no clear difference** |
| N kv_lif − kv_pop_lif @0.003 | -0.16 / +3.40 / -0.18 / +1.05 / +2.73 | +1.37 | 3/5 | **no clear difference** |
| N q_lif_mem − q_pop_lif_mem @0.003 | -0.96 / -1.76 / +1.48 / -1.16 / -3.52 | -1.19 | 1/5 | **pop LIF better** |
| N q_flif_mem − q_pop_lif_mem @0.003 | -0.52 / -0.78 / +1.11 / -0.24 / -2.54 | -0.59 | 1/5 | **no clear difference** |
| N q_lif − q_pop_lif @0.001 (report) | +0.73 / -3.61 / -0.47 / +0.13 / +0.10 | -0.62 | 3/5 | no clear difference |

**L reading (q_lif):** not supported.

**Reading per model (pre-registered):** q_lif: neither; q_lif_mem: neither; q_lif_acc: the Neocortex order / memory now counts, but Spikformer(DE) is not beaten; q_lif_mem_acc: does not beat Spikformer(DE); q_flif: does not beat Spikformer(DE); q_flif_mem: does not beat Spikformer(DE)
