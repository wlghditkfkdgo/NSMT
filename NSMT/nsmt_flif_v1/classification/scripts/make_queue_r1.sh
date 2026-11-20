#!/usr/bin/env bash
# MODI 2026-11-20: regenerates the (git-ignored) queue of hyperparameter range 1 (docs/MODI_PROJECT_LOG.md 2026-11-20; user decision
# 11-20 "4-1 검증(작음) -> 범위 1(이후 모든 비교의 학습 조건을 정함) -> 4-2 (가) + (나)"). 45 runs, default PE=0 (no relative
# position bias), every neuron SpikeDE, no aux loss -- the bias-off reference (flif-v1-ett-nope / flif-v1-mitbih-nope) plus
#   WDX=1 (--wd_exclude_1d 1: no weight decay on the BatchNorm weights / biases) and, ETT only, VPE=4 (--val_per_epoch 4)
#   suite flif-v1-ett-r1 (36 runs): spkall_lif (run_ett_mixer.sh), nsmt4q_pop_<lif|flif>_msn_sde (run_ett_dual.sh), ETTh2 / ETTm2 x H96 / H720 x seeds 42 / 7 / 2026
#   suite flif-v1-mitbih-r1 (9 runs): the same three conditions (run_mitbih4.sh), seeds 42 / 7 / 2026, last-epoch weights
# Order: ETTm2 (longest), MIT-BIH, ETTh2. NSMT/: bash scripts/gpu_pool2.sh <queue> <gpus> (appends the gpu).
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"; ET=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh; MX=nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh; M4=nsmt_flif_v1/classification/scripts/run_mitbih4.sh
Q=$QD/r1-20261120.txt; : > $Q
ett() { for pl in 720 96; do for s in $SEEDS; do
  echo "COND=spkall_lif SUITE=flif-v1-ett-r1 WDX=1 VPE=4 bash $MX $1 $pl $s"
  for n in lif flif; do echo "COND=nsmt4q_pop_${n}_msn_sde SUITE=flif-v1-ett-r1 WDX=1 VPE=4 bash $ET $1 $pl $s"; done; done; done >> $Q; }
ett ETTm2
for s in $SEEDS; do for c in spkall_lif nsmt4q_pop_lif_msn_sde nsmt4q_pop_flif_msn_sde; do echo "COND=$c SUITE=flif-v1-mitbih-r1 WDX=1 bash $M4 $s"; done; done >> $Q
ett ETTh2
echo "$Q: $(grep -c . $Q) jobs"
