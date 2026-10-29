# Whole-model neuron swap batch — summary

See the module docstring for names and rules. Exploratory (one CinC fold, 3 seeds).

## ETT forecasting (test MSE; r = 100 (A − B) / B, %)

| question | A | B | cells b / . / w | R | pairs A < B | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|---|
| W1 every layer vs neocortex only, LIF | mixall_lif_bidir | mix_lif_bidir | | | | incomplete | | |
| W1 every layer vs neocortex only, f-LIF | mixall_flif_bidir | mix_flif_bidir | | | | incomplete | | |
| W1 every layer vs neocortex only, pop LIF | mixall_pop_lif_bidir | mix_pop_lif_bidir | | | | incomplete | | |
| W1 every layer vs neocortex only, pop f-LIF | mixall_pop_flif_bidir | mix_pop_flif_bidir | | | | incomplete | | |
| W2 Neo-mixer vs Spikformer-style, both every layer, LIF (criterion 2) | mixall_lif_bidir | spkall_lif | | | | incomplete | | |
| W2 Neo-mixer vs Spikformer-style, both every layer, f-LIF (criterion 2) | mixall_flif_bidir | spkall_flif | | | | incomplete | | |
| W2 Neo-mixer vs Spikformer-style, both every layer, pop LIF (criterion 2) | mixall_pop_lif_bidir | spkall_pop_lif | | | | incomplete | | |
| W2 Neo-mixer vs Spikformer-style, both every layer, pop f-LIF (criterion 2) | mixall_pop_flif_bidir | spkall_pop_flif | | | | incomplete | | |
| W3 (report) Spikformer-style every layer vs spikingjelly, LIF | spkall_lif | spk_ssa | | | | incomplete | | |
| W3 (report) Spikformer-style every layer vs spikingjelly, f-LIF | spkall_flif | spk_ssa | | | | incomplete | | |
| W3 (report) Spikformer-style every layer vs spikingjelly, pop LIF | spkall_pop_lif | spk_ssa | | | | incomplete | | |
| W3 (report) Spikformer-style every layer vs spikingjelly, pop f-LIF | spkall_pop_flif | spk_ssa | | | | incomplete | | |
| W4 fractional order, Neo-mixer every layer (15 cells) | mixall_flif_bidir | mixall_lif_bidir | | | | incomplete | | |
| W4 fractional order, Spikformer-style every layer (15 cells) | spkall_flif | spkall_lif | | | | incomplete | | |
| W5 (report) Neo-mixer every layer vs Spikformer-style, LIF (criterion 2) | mixall_lif_bidir | spk_ssa | | | | incomplete | | |
| W5 (report) Neo-mixer every layer vs Spikformer-style, f-LIF (criterion 2) | mixall_flif_bidir | spk_ssa | | | | incomplete | | |
| W5 (report) Neo-mixer every layer vs Spikformer-style, pop LIF (criterion 2) | mixall_pop_lif_bidir | spk_ssa | | | | incomplete | | |
| W5 (report) Neo-mixer every layer vs Spikformer-style, pop f-LIF (criterion 2) | mixall_pop_flif_bidir | spk_ssa | | | | incomplete | | |

### ETTh2 H96 (quick cell; better / worse need 3/3 seeds)

| question | A | B | r per seed | R | verdict | energy A/B |
|---|---|---|---|---|---|---|
| C1 implementation: spikeDE LIF outside the mixer, LIF | mixall_lif_bidir | mix_lif_bidir | +2.41 / +0.64 / +2.13 | +1.73 | worse | 1.01 |
| C1 implementation: spikeDE LIF outside the mixer, f-LIF | mixlb_flif_bidir | mix_flif_bidir | +3.24 / +2.58 / +0.54 | +2.12 | worse | 1.01 |
| C1 implementation: spikeDE LIF outside the mixer, pop LIF | mixlb_pop_lif_bidir | mix_pop_lif_bidir | -0.60 / +1.26 / +2.47 | +1.04 | no clear difference | 1.00 |
| C1 implementation: spikeDE LIF outside the mixer, pop f-LIF | mixlb_pop_flif_bidir | mix_pop_flif_bidir | +1.59 / +2.17 / +3.15 | +2.30 | worse | 1.00 |
| C2 neuron type outside the mixer, f-LIF | mixall_flif_bidir | mixlb_flif_bidir | -1.75 / -1.87 / -2.10 | -1.91 | better | 1.21 |
| C2 neuron type outside the mixer, pop LIF | mixall_pop_lif_bidir | mixlb_pop_lif_bidir | +4.74 / +2.12 / +1.80 | +2.89 | worse | 1.54 |
| C2 neuron type outside the mixer, pop f-LIF | mixall_pop_flif_bidir | mixlb_pop_flif_bidir | -1.15 / -0.06 / -1.81 | -1.01 | better | 1.85 |
| Q3 Spikformer-style every layer vs spikingjelly, LIF | spkall_lif | spk_ssa | -1.16 / +1.91 / +1.08 | +0.61 | no clear difference | 1.00 |
| Q3 Spikformer-style every layer vs spikingjelly, f-LIF | spkall_flif | spk_ssa | +0.28 / +1.66 / +1.91 | +1.29 | worse | 1.02 |
| Q3 Spikformer-style every layer vs spikingjelly, pop LIF | spkall_pop_lif | spk_ssa | -2.42 / -1.24 / -0.56 | -1.40 | better | 2.39 |
| Q3 Spikformer-style every layer vs spikingjelly, pop f-LIF | spkall_pop_flif | spk_ssa | -1.10 / -0.10 / -0.13 | -0.44 | no clear difference | 3.52 |

## MIT-BIH (test macro-F1 %, last-epoch weights)

| question | A | B | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|---|---|
| W1 every layer vs neocortex only, LIF | mixall_lif_bidir | mix_lif_bidir | | | | | incomplete | | |
| W1 every layer vs neocortex only, f-LIF | mixall_flif_bidir | mix_flif_bidir | | | | | incomplete | | |
| W1 every layer vs neocortex only, pop LIF | mixall_pop_lif_bidir | mix_pop_lif_bidir | | | | | incomplete | | |
| W1 every layer vs neocortex only, pop f-LIF | mixall_pop_flif_bidir | mix_pop_flif_bidir | | | | | incomplete | | |
| W2 Neo-mixer vs Spikformer-style, both every layer, LIF (criterion 2) | mixall_lif_bidir | spkall_lif | | | | | incomplete | | |
| W2 Neo-mixer vs Spikformer-style, both every layer, f-LIF (criterion 2) | mixall_flif_bidir | spkall_flif | | | | | incomplete | | |
| W2 Neo-mixer vs Spikformer-style, both every layer, pop LIF (criterion 2) | mixall_pop_lif_bidir | spkall_pop_lif | | | | | incomplete | | |
| W2 Neo-mixer vs Spikformer-style, both every layer, pop f-LIF (criterion 2) | mixall_pop_flif_bidir | spkall_pop_flif | | | | | incomplete | | |
| W3 (report) Spikformer-style every layer vs spikingjelly, LIF | spkall_lif | spk_ssa | | | | | incomplete | | |
| W3 (report) Spikformer-style every layer vs spikingjelly, f-LIF | spkall_flif | spk_ssa | | | | | incomplete | | |
| W3 (report) Spikformer-style every layer vs spikingjelly, pop LIF | spkall_pop_lif | spk_ssa | | | | | incomplete | | |
| W3 (report) Spikformer-style every layer vs spikingjelly, pop f-LIF | spkall_pop_flif | spk_ssa | | | | | incomplete | | |
| W4 fractional order, Neo-mixer every layer | mixall_flif_bidir | mixall_lif_bidir | | | | | incomplete | | |
| W4 fractional order, Spikformer-style every layer | spkall_flif | spkall_lif | | | | | incomplete | | |
| W5 (report) Neo-mixer every layer vs Spikformer-style, LIF (criterion 2) | mixall_lif_bidir | spk_ssa | | | | | incomplete | | |
| W5 (report) Neo-mixer every layer vs Spikformer-style, f-LIF (criterion 2) | mixall_flif_bidir | spk_ssa | | | | | incomplete | | |
| W5 (report) Neo-mixer every layer vs Spikformer-style, pop LIF (criterion 2) | mixall_pop_lif_bidir | spk_ssa | | | | | incomplete | | |
| W5 (report) Neo-mixer every layer vs Spikformer-style, pop f-LIF (criterion 2) | mixall_pop_flif_bidir | spk_ssa | | | | | incomplete | | |
| C1 implementation: spikeDE LIF outside the mixer, LIF | mixall_lif_bidir | mix_lif_bidir | | | | | incomplete | | |
| C1 implementation: spikeDE LIF outside the mixer, f-LIF | mixlb_flif_bidir | mix_flif_bidir | | | | | incomplete | | |
| C1 implementation: spikeDE LIF outside the mixer, pop LIF | mixlb_pop_lif_bidir | mix_pop_lif_bidir | | | | | incomplete | | |
| C1 implementation: spikeDE LIF outside the mixer, pop f-LIF | mixlb_pop_flif_bidir | mix_pop_flif_bidir | | | | | incomplete | | |
| C2 neuron type outside the mixer, f-LIF | mixall_flif_bidir | mixlb_flif_bidir | | | | | incomplete | | |
| C2 neuron type outside the mixer, pop LIF | mixall_pop_lif_bidir | mixlb_pop_lif_bidir | | | | | incomplete | | |
| C2 neuron type outside the mixer, pop f-LIF | mixall_pop_flif_bidir | mixlb_pop_flif_bidir | | | | | incomplete | | |

## CinC 2017 fold 0, 90 epochs (test score %, best-validation weights)

| question | A | B | A (mean) | B (mean) | A − B per seed (%p) | mean | verdict | energy A/B | criterion (2) |
|---|---|---|---|---|---|---|---|---|---|
| D1w T = N neocortex vs memoryless, every layer LIF (main) | spkneoall_lif | spkneoall_lif_nomem | | | | | incomplete | | |
| D1w T = N neocortex vs memoryless, every layer f-LIF (main) | spkneoall_flif | spkneoall_flif_nomem | | | | | incomplete | | |
| D1w T = N neocortex vs memoryless, every layer pop LIF (main) | spkneoall_pop_lif | spkneoall_pop_lif_nomem | | | | | incomplete | | |
| D1w T = N neocortex vs memoryless, every layer pop f-LIF (main) | spkneoall_pop_flif | spkneoall_pop_flif_nomem | | | | | incomplete | | |
| D2w B form vs Spikformer-style, both every layer, LIF | spkneoall_lif | spkall_lif | | | | | incomplete | | |
| D2w B form vs Spikformer-style, both every layer, f-LIF | spkneoall_flif | spkall_flif | | | | | incomplete | | |
| D2w B form vs Spikformer-style, both every layer, pop LIF | spkneoall_pop_lif | spkall_pop_lif | | | | | incomplete | | |
| D2w B form vs Spikformer-style, both every layer, pop f-LIF | spkneoall_pop_flif | spkall_pop_flif | | | | | incomplete | | |
| D3w (report) B form every layer vs neocortex only, LIF | spkneoall_lif | spkneo_lif | | | | | incomplete | | |
| D3w (report) B form every layer vs neocortex only, f-LIF | spkneoall_flif | spkneo_flif | | | | | incomplete | | |
| D3w (report) B form every layer vs neocortex only, pop LIF | spkneoall_pop_lif | spkneo_pop_lif | | | | | incomplete | | |
| D3w (report) B form every layer vs neocortex only, pop f-LIF | spkneoall_pop_flif | spkneo_pop_flif | | | | | incomplete | | |
| D3w (report) Spikformer-style every layer vs spikingjelly, LIF | spkall_lif | spk_ssa | | | | | incomplete | | |
| D3w (report) Spikformer-style every layer vs spikingjelly, f-LIF | spkall_flif | spk_ssa | | | | | incomplete | | |
| D3w (report) Spikformer-style every layer vs spikingjelly, pop LIF | spkall_pop_lif | spk_ssa | | | | | incomplete | | |
| D3w (report) Spikformer-style every layer vs spikingjelly, pop f-LIF | spkall_pop_flif | spk_ssa | | | | | incomplete | | |
| D3w (report) memoryless control every layer vs neocortex-only LIF control, LIF | spkneoall_lif_nomem | spkneo_lif_nomem | | | | | incomplete | | |
| D3w (report) memoryless control every layer vs neocortex-only LIF control, f-LIF | spkneoall_flif_nomem | spkneo_lif_nomem | | | | | incomplete | | |
| D3w (report) memoryless control every layer vs neocortex-only LIF control, pop LIF | spkneoall_pop_lif_nomem | spkneo_lif_nomem | | | | | incomplete | | |
| D3w (report) memoryless control every layer vs neocortex-only LIF control, pop f-LIF | spkneoall_pop_flif_nomem | spkneo_lif_nomem | | | | | incomplete | | |
