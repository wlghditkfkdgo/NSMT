#!/usr/bin/env bash
# MODI 2026-11-08: regenerates the (git-ignored) queue of the membrane shortcut test (docs/MODI_PROJECT_LOG.md 2026-11-08; user request
# 11-08 "Residual Connection을 Membrane Shortcut으로 변경 / Neocortex 레이어끼리 membrane shortcut"):
#   nsmt4q_pop_<lif|flif>_<msh|msn|mshn> x seeds 42 / 7 / 2026
#     ETT: ETTh2 / ETTm2 x H96 / H720, unchanged run_ett_dual.sh settings -- suite flif-v1-ett-ms (72 runs; baselines flif-v1-ett-query
#          (ETTh2) + flif-v1-ett-query-m2 (ETTm2))
#     MIT-BIH: unchanged run_mitbih4.sh defaults (lr 0.01, full training file, last epoch tested) -- suite flif-v1-mitbih-ms (18 runs;
#          baseline flif-v1-mitbih-query)
# Order: every pop LIF run first (the model of the request), then pop f-LIF; inside each, ETTm2 H720, ETTm2 H96, MIT-BIH, ETTh2.
# NSMT/: bash scripts/gpu_pool2.sh <queue> <gpus> (appends the gpu).
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"; MB=nsmt_flif_v1/classification/scripts/run_mitbih4.sh; ET=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh
Q=$QD/ms-20261108.txt; : > $Q
for n in pop_lif pop_flif; do
  for pl in 720 96; do for v in msh msn mshn; do for s in $SEEDS; do
    echo "COND=nsmt4q_${n}_${v} SUITE=flif-v1-ett-ms bash $ET ETTm2 $pl $s"; done; done; done >> $Q
  for v in msh msn mshn; do for s in $SEEDS; do
    echo "COND=nsmt4q_${n}_${v} SUITE=flif-v1-mitbih-ms bash $MB $s"; done; done >> $Q
  for pl in 96 720; do for v in msh msn mshn; do for s in $SEEDS; do
    echo "COND=nsmt4q_${n}_${v} SUITE=flif-v1-ett-ms bash $ET ETTh2 $pl $s"; done; done; done >> $Q
done
echo "$Q: $(grep -c . $Q) jobs"
