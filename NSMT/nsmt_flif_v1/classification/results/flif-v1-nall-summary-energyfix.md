# Whole-model neuron swap batch — summary

See the module docstring for names and rules. Exploratory (one CinC fold, 3 seeds).

## ETT forecasting (test MSE; r = 100 (A − B) / B, %) — Spikformer-style energy recounted

| question | A | B | cells b / . / w | R | pairs A < B | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|---|
| W1 every layer vs neocortex only, LIF | mixall_lif_bidir | mix_lif_bidir | | | | incomplete | | |
| W1 every layer vs neocortex only, f-LIF | mixall_flif_bidir | mix_flif_bidir | | | | incomplete | | |
| W1 every layer vs neocortex only, pop LIF | mixall_pop_lif_bidir | mix_pop_lif_bidir | | | | incomplete | | |
| W1 every layer vs neocortex only, pop f-LIF | mixall_pop_flif_bidir | mix_pop_flif_bidir | | | | incomplete | | |
| W4 fractional order, Neo-mixer every layer (3 cells) | mixall_flif_bidir | mixall_lif_bidir | | | | incomplete | | |
| W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), LIF (criterion 2) | mixall_lif_bidir | spk_ssa | | | | incomplete | | |
| W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), f-LIF (criterion 2) | mixall_flif_bidir | spk_ssa | | | | incomplete | | |
| W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), pop LIF (criterion 2) | mixall_pop_lif_bidir | spk_ssa | | | | incomplete | | |
| W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), pop f-LIF (criterion 2) | mixall_pop_flif_bidir | spk_ssa | | | | incomplete | | |

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
| W1 every layer vs neocortex only, LIF | mixall_lif_bidir | mix_lif_bidir | | | | | incomplete | | |
| W1 every layer vs neocortex only, f-LIF | mixall_flif_bidir | mix_flif_bidir | | | | | incomplete | | |
| W1 every layer vs neocortex only, pop LIF | mixall_pop_lif_bidir | mix_pop_lif_bidir | | | | | incomplete | | |
| W1 every layer vs neocortex only, pop f-LIF | mixall_pop_flif_bidir | mix_pop_flif_bidir | | | | | incomplete | | |
| W4 fractional order, Neo-mixer every layer | mixall_flif_bidir | mixall_lif_bidir | | | | | incomplete | | |
| W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), LIF (criterion 2) | mixall_lif_bidir | spk_ssa | | | | | incomplete | | |
| W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), f-LIF (criterion 2) | mixall_flif_bidir | spk_ssa | | | | | incomplete | | |
| W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), pop LIF (criterion 2) | mixall_pop_lif_bidir | spk_ssa | | | | | incomplete | | |
| W5 Neo-mixer every layer vs Spikformer-style (spikingjelly LIF), pop f-LIF (criterion 2) | mixall_pop_flif_bidir | spk_ssa | | | | | incomplete | | |
| C1 implementation: spikeDE LIF outside the mixer, LIF | mixall_lif_bidir | mix_lif_bidir | | | | | incomplete | | |
| C1 implementation: spikeDE LIF outside the mixer, f-LIF | mixlb_flif_bidir | mix_flif_bidir | | | | | incomplete | | |
| C1 implementation: spikeDE LIF outside the mixer, pop LIF | mixlb_pop_lif_bidir | mix_pop_lif_bidir | | | | | incomplete | | |
| C1 implementation: spikeDE LIF outside the mixer, pop f-LIF | mixlb_pop_flif_bidir | mix_pop_flif_bidir | | | | | incomplete | | |
| C2 neuron type outside the mixer, f-LIF | mixall_flif_bidir | mixlb_flif_bidir | | | | | incomplete | | |
| C2 neuron type outside the mixer, pop LIF | mixall_pop_lif_bidir | mixlb_pop_lif_bidir | | | | | incomplete | | |
| C2 neuron type outside the mixer, pop f-LIF | mixall_pop_flif_bidir | mixlb_pop_flif_bidir | | | | | incomplete | | |

## CinC 2017

Dropped on 2026-10-29 17:1x (user): the B form is a dual-path model, so its neuron conditions swap only the neocortex -- the existing spkneo_<n> runs (flif-v1-rhythm-addneo90-cinc). The 8 B form every-layer runs were stopped at epoch 34-44 of 90 (SKIP rc=143 in flif-v1-rhythm-nall-cinc/raw.txt) and are not used.
