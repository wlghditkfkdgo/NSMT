#!/usr/bin/env bash
# MODI 2026-11-21: regenerates the (git-ignored) queue of 4-2, the coupling placement test (docs/MODI_PROJECT_LOG.md 2026-11-21; user
# decision 11-20 "4-1 검증(작음) -> 범위 1(이후 모든 비교의 학습 조건을 정함) -> 4-2 (가) + (나)"). 60 runs at the runner defaults
# (range-1 training, no relative position bias, every neuron SpikeDE, no aux loss):
#   (가) nsmt4q_pop_<lif|flif>_msn_sde_nc   (Neocortex output x per-channel alpha into the hippocampus embedding current; attention kept)
#   (나) nsmt4q_pop_<lif|flif>_msn_sde_ncna (the same without the hippocampus attention)
#   suite flif-v1-ett-nc (48 runs): run_ett_dual.sh, ETTh2 / ETTm2 x H96 / H720 x seeds 42 / 7 / 2026
#   suite flif-v1-mitbih-nc (12 runs): run_mitbih4.sh, seeds 42 / 7 / 2026, last-epoch weights
# The base and Spikformer(DE) are the existing range-1 runs (flif-v1-ett-r1 / flif-v1-mitbih-r1); none are rerun.
# Order: ETTm2 (longest), MIT-BIH, ETTh2. NSMT/: bash scripts/gpu_pool2.sh <queue> <gpus> (appends the gpu).
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"; ET=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh; M4=nsmt_flif_v1/classification/scripts/run_mitbih4.sh
CONDS="nsmt4q_pop_lif_msn_sde_nc nsmt4q_pop_flif_msn_sde_nc nsmt4q_pop_lif_msn_sde_ncna nsmt4q_pop_flif_msn_sde_ncna"
Q=$QD/nc-20261121.txt; : > $Q
ett() { for pl in 720 96; do for s in $SEEDS; do for c in $CONDS; do echo "COND=$c SUITE=flif-v1-ett-nc bash $ET $1 $pl $s"; done; done; done >> $Q; }
ett ETTm2
for s in $SEEDS; do for c in $CONDS; do echo "COND=$c SUITE=flif-v1-mitbih-nc bash $M4 $s"; done; done >> $Q
ett ETTh2
echo "$Q: $(grep -c . $Q) jobs"
