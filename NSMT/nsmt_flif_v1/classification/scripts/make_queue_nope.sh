#!/usr/bin/env bash
# MODI 2026-11-19: regenerates the (git-ignored) queue of the positional-encoding-off test (docs/MODI_PROJECT_LOG.md; user request 11-19
# "relative positional encoding을 끄고 실험을 해줘. (Spikformer 도 마찬가지)"). PE=0 = --ssa_pe 0 (no relative position bias in the attention),
# every neuron SpikeDE, current training setting, no aux loss:
#   suite flif-v1-ett-nope (36 runs): spkall_lif (run_ett_mixer.sh), nsmt4q_pop_<lif|flif>_msn_sde (run_ett_dual.sh), ETTh2 / ETTm2 x H96 / H720 x seeds 42 / 7 / 2026
#   suite flif-v1-mitbih-nope (9 runs): the same three conditions (run_mitbih4.sh), seeds 42 / 7 / 2026
# The bias-on arms are the existing runs of flif-v1-ett-sde / flif-v1-mitbih-sde. Order: ETTm2 (longest), MIT-BIH, ETTh2.
# NSMT/: bash scripts/gpu_pool2.sh <queue> <gpus> (appends the gpu).
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"; ET=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh; MX=nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh; M4=nsmt_flif_v1/classification/scripts/run_mitbih4.sh
Q=$QD/nope-20261119.txt; : > $Q
ett() { for pl in 720 96; do for s in $SEEDS; do
  echo "COND=spkall_lif SUITE=flif-v1-ett-nope PE=0 bash $MX $1 $pl $s"
  for n in lif flif; do echo "COND=nsmt4q_pop_${n}_msn_sde SUITE=flif-v1-ett-nope PE=0 bash $ET $1 $pl $s"; done; done; done >> $Q; }
ett ETTm2
for s in $SEEDS; do for c in spkall_lif nsmt4q_pop_lif_msn_sde nsmt4q_pop_flif_msn_sde; do echo "COND=$c SUITE=flif-v1-mitbih-nope PE=0 bash $M4 $s"; done; done >> $Q
ett ETTh2
echo "$Q: $(grep -c . $Q) jobs"
