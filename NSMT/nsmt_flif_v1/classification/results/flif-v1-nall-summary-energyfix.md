# Whole-model neuron swap batch — summary

See the module docstring for names and rules. Exploratory (one CinC fold, 3 seeds).

## ETT forecasting (test MSE; r = 100 (A − B) / B, %) — Spikformer-style energy recounted

| question | A | B | cells b / . / w | R | pairs A < B | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|---|
| W1 every layer vs neocortex only, LIF | mixall_lif_bidir | mix_lif_bidir | 0 / 2 / 2 | +0.80 | 4/12 | no clear difference | 1.01 |  |
| W1 every layer vs neocortex only, f-LIF | mixall_flif_bidir | mix_flif_bidir | 0 / 4 / 0 | +0.17 | 3/12 | no clear difference | 1.18 |  |
| W1 every layer vs neocortex only, pop LIF | mixall_pop_lif_bidir | mix_pop_lif_bidir | 0 / 2 / 2 | +1.48 | 5/12 | no clear difference | 1.43 |  |
| W1 every layer vs neocortex only, pop f-LIF | mixall_pop_flif_bidir | mix_pop_flif_bidir | 0 / 3 / 1 | +0.44 | 6/12 | no clear difference | 1.68 |  |
| W4 fractional order, Neo-mixer every layer (3 cells) | mixall_flif_bidir | mixall_lif_bidir | 1 / 2 / 0 | -0.34 | 5/9 | no clear difference | 1.42 |  |
| W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), LIF (criterion 2) | mixall_lif_bidir | spk_ssa | 0 / 2 / 2 | +1.02 | 3/12 | worse | 0.21 | not met |
| W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), f-LIF (criterion 2) | mixall_flif_bidir | spk_ssa | 0 / 4 / 0 | +0.44 | 3/12 | no clear difference | 0.29 | met |
| W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), pop LIF (criterion 2) | mixall_pop_lif_bidir | spk_ssa | 0 / 2 / 2 | +1.95 | 2/12 | worse | 0.55 | not met |
| W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), pop f-LIF (criterion 2) | mixall_pop_flif_bidir | spk_ssa | 0 / 3 / 1 | +0.40 | 6/12 | no clear difference | 1.17 | not met |

### ETTh2 H96 (quick cell; better / worse need 3/3 seeds; Q3 = report since 10-29 17:1x)

| question | A | B | r per seed | R | verdict | energy A/B |
|---|---|---|---|---|---|---|
| C1 implementation: spikeDE LIF outside the mixer, LIF | mixall_lif_bidir | mix_lif_bidir | +2.41 / +0.64 / +2.13 | +1.73 | worse | 1.01 |
| C1 implementation: spikeDE LIF outside the mixer, f-LIF | mixlb_flif_bidir | mix_flif_bidir | +3.24 / +2.58 / +0.54 | +2.12 | worse | 1.01 |
| C1 implementation: spikeDE LIF outside the mixer, pop LIF | mixlb_pop_lif_bidir | mix_pop_lif_bidir | -0.60 / +1.26 / +2.47 | +1.04 | no clear difference | 1.00 |
| C1 implementation: spikeDE LIF outside the mixer, pop f-LIF | mixlb_pop_flif_bidir | mix_pop_flif_bidir | +1.59 / +2.17 / +3.15 | +2.30 | worse | 1.00 |
| C2 neuron type outside the mixer, f-LIF | mixall_flif_bidir | mixlb_flif_bidir | -1.75 / -1.87 / -2.10 | -1.91 | better | 1.21 |
| C2 neuron type outside the mixer, pop LIF | mixall_pop_lif_bidir | mixlb_pop_lif_bidir | +4.74 / +2.12 / +1.80 | +2.89 | worse | 1.54 |
| C2 neuron type outside the mixer, pop f-LIF | mixall_pop_flif_bidir | mixlb_pop_flif_bidir | -1.15 / -0.06 / -1.81 | -1.01 | better | 1.85 |
| Q3 Spikformer-style every layer vs spikingjelly, LIF | spkall_lif | spk_ssa | -1.16 / +1.91 / +1.08 | +0.61 | no clear difference | 1.01 |
| Q3 Spikformer-style every layer vs spikingjelly, f-LIF | spkall_flif | spk_ssa | +0.28 / +1.66 / +1.91 | +1.29 | worse | 1.02 |
| Q3 Spikformer-style every layer vs spikingjelly, pop LIF | spkall_pop_lif | spk_ssa | -2.42 / -1.24 / -0.56 | -1.40 | better | 3.05 |
| Q3 Spikformer-style every layer vs spikingjelly, pop f-LIF | spkall_pop_flif | spk_ssa | -1.10 / -0.10 / -0.13 | -0.44 | no clear difference | 4.73 |

## MIT-BIH (test macro-F1 %, last-epoch weights)

| question | A | B | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|---|---|
| W1 every layer vs neocortex only, LIF | mixall_lif_bidir | mix_lif_bidir | 90.70 | 91.14 | -0.36 / -0.58 / -0.39 | -0.44 | no clear difference | 1.03 |  |
| W1 every layer vs neocortex only, f-LIF | mixall_flif_bidir | mix_flif_bidir | 91.12 | 91.13 | -0.11 / +0.08 / -0.00 | -0.01 | no clear difference | 1.06 |  |
| W1 every layer vs neocortex only, pop LIF | mixall_pop_lif_bidir | mix_pop_lif_bidir | 88.84 | 91.27 | -2.47 / -2.52 / -2.31 | -2.44 | worse | 1.24 |  |
| W1 every layer vs neocortex only, pop f-LIF | mixall_pop_flif_bidir | mix_pop_flif_bidir | 89.70 | 91.19 | -1.63 / -1.44 / -1.40 | -1.49 | worse | 1.39 |  |
| W4 fractional order, Neo-mixer every layer | mixall_flif_bidir | mixall_lif_bidir | 91.12 | 90.70 | +0.50 / +0.34 / +0.42 | +0.42 | no clear difference | 1.07 |  |
| W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), LIF (criterion 2) | mixall_lif_bidir | spk_ssa | 90.70 | 92.22 | -1.71 / -1.60 / -1.26 | -1.52 | worse | 0.23 | not met |
| W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), f-LIF (criterion 2) | mixall_flif_bidir | spk_ssa | 91.12 | 92.22 | -1.21 / -1.26 / -0.84 | -1.10 | worse | 0.25 | not met |
| W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), pop LIF (criterion 2) | mixall_pop_lif_bidir | spk_ssa | 88.84 | 92.22 | -3.48 / -3.33 / -3.34 | -3.39 | worse | 0.39 | not met |
| W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), pop f-LIF (criterion 2) | mixall_pop_flif_bidir | spk_ssa | 89.70 | 92.22 | -2.47 / -2.99 / -2.11 | -2.52 | worse | 0.64 | not met |
| C1 implementation: spikeDE LIF outside the mixer, LIF | mixall_lif_bidir | mix_lif_bidir | 90.70 | 91.14 | -0.36 / -0.58 / -0.39 | -0.44 | no clear difference | 1.03 |  |
| C1 implementation: spikeDE LIF outside the mixer, f-LIF | mixlb_flif_bidir | mix_flif_bidir | 90.39 | 91.13 | -1.26 / -0.14 / -0.83 | -0.74 | worse | 1.03 |  |
| C1 implementation: spikeDE LIF outside the mixer, pop LIF | mixlb_pop_lif_bidir | mix_pop_lif_bidir | 90.83 | 91.27 | -0.73 / -0.40 / -0.20 | -0.44 | no clear difference | 0.99 |  |
| C1 implementation: spikeDE LIF outside the mixer, pop f-LIF | mixlb_pop_flif_bidir | mix_pop_flif_bidir | 90.64 | 91.19 | -1.03 / +0.08 / -0.67 | -0.54 | worse | 1.00 |  |
| C2 neuron type outside the mixer, f-LIF | mixall_flif_bidir | mixlb_flif_bidir | 91.12 | 90.39 | +1.15 / +0.22 / +0.83 | +0.73 | better | 1.02 |  |
| C2 neuron type outside the mixer, pop LIF | mixall_pop_lif_bidir | mixlb_pop_lif_bidir | 88.84 | 90.83 | -1.75 / -2.12 / -2.12 | -1.99 | worse | 1.25 |  |
| C2 neuron type outside the mixer, pop f-LIF | mixall_pop_flif_bidir | mixlb_pop_flif_bidir | 89.70 | 90.64 | -0.59 / -1.52 / -0.73 | -0.95 | worse | 1.39 |  |

## CinC 2017

Dropped on 2026-10-29 17:1x (user): the B form is a dual-path model, so its neuron conditions swap only the neocortex -- the existing spkneo_<n> runs (flif-v1-rhythm-addneo90-cinc). The 8 B form every-layer runs were stopped at epoch 34-44 of 90 (SKIP rc=143 in flif-v1-rhythm-nall-cinc/raw.txt) and are not used.
