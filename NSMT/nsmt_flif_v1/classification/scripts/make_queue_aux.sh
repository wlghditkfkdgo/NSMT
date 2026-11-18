#!/usr/bin/env bash
# MODI 2026-11-19: regenerates the (git-ignored) queue of the aux-loss test (docs/MODI_PROJECT_LOG.md; user request 11-19 "1. auxiliary
# loss를 ON해서 실험하여 비교해줘"). NSMT's reconstruction loss on the msn base, both bases (spikingjelly hippocampus / every neuron
# SpikeDE), pop LIF and pop f-LIF:
#   suite flif-v1-ett-aux (48 runs): nsmt4q_pop_<lif|flif>_{msn,msn_sde}_aux (run_ett_dual.sh: ETTh2 alpha 0.8 keep 0.3, ETTm2 alpha 0.5
#         keep 0.25, r = alpha * H / 720); ETTh2 / ETTm2 x H96 / H720, seeds 42 / 7 / 2026, input 96, everything else unchanged
#   suite flif-v1-mitbih-aux (12 runs): the same four conditions (run_mitbih4.sh, AUX 0.1, keep 0.15); seeds 42 / 7 / 2026
# The aux-off arms and Spikformer-style are the existing runs (flif-v1-ett-ms / -sde, flif-v1-gen-ett, flif-v1-mitbih-ms / -sde,
# flif-v1-gen-mitbih). Order: ETTm2 (longest), MIT-BIH, ETTh2. NSMT/: bash scripts/gpu_pool2.sh <queue> <gpus> (appends the gpu).
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"; ET=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh; M4=nsmt_flif_v1/classification/scripts/run_mitbih4.sh
Q=$QD/aux-20261119.txt; : > $Q
CONDS="nsmt4q_pop_lif_msn_aux nsmt4q_pop_flif_msn_aux nsmt4q_pop_lif_msn_sde_aux nsmt4q_pop_flif_msn_sde_aux"
ett() { for pl in 720 96; do for s in $SEEDS; do for c in $CONDS; do echo "COND=$c SUITE=flif-v1-ett-aux bash $ET $1 $pl $s"; done; done; done >> $Q; }
ett ETTm2
for s in $SEEDS; do for c in $CONDS; do echo "COND=$c SUITE=flif-v1-mitbih-aux bash $M4 $s"; done; done >> $Q
ett ETTh2
echo "$Q: $(grep -c . $Q) jobs"
