#!/usr/bin/env bash
# MODI 2026-11-13: regenerates the (git-ignored) queue of the frequency-band test (docs/MODI_PROJECT_LOG.md 2026-11-13; user request
# 11-13 "주파수 분석을 해서 4개의 요소를 갖도록 한 다음 이것을 population(k=4) LIF neocortex에 추가"; base = Neocortex MS, user decision 11-13):
#   nsmt4q_pop_<lif|flif>_msn_<freq|same|rev> x seeds 42 / 7 / 2026
#     ETT: ETTh2 / ETTm2 x H96 / H720, unchanged run_ett_dual.sh settings -- suite flif-v1-ett-band (72 runs; baseline = the msn runs
#          of flif-v1-ett-ms)
#     MIT-BIH: unchanged run_mitbih4.sh defaults (lr 0.01, full training file, last epoch tested) -- suite flif-v1-mitbih-band (18 runs;
#          baseline = the msn runs of flif-v1-mitbih-ms)
# Order: every pop LIF run first, then pop f-LIF; inside each, ETTm2 H720, ETTm2 H96, MIT-BIH, ETTh2.
# NSMT/: bash scripts/gpu_pool2.sh <queue> <gpus> (appends the gpu).
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"; MB=nsmt_flif_v1/classification/scripts/run_mitbih4.sh; ET=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh
Q=$QD/band-20261113.txt; : > $Q
for n in pop_lif pop_flif; do
  for pl in 720 96; do for v in freq same rev; do for s in $SEEDS; do
    echo "COND=nsmt4q_${n}_msn_${v} SUITE=flif-v1-ett-band bash $ET ETTm2 $pl $s"; done; done; done >> $Q
  for v in freq same rev; do for s in $SEEDS; do
    echo "COND=nsmt4q_${n}_msn_${v} SUITE=flif-v1-mitbih-band bash $MB $s"; done; done >> $Q
  for pl in 96 720; do for v in freq same rev; do for s in $SEEDS; do
    echo "COND=nsmt4q_${n}_msn_${v} SUITE=flif-v1-ett-band bash $ET ETTh2 $pl $s"; done; done; done >> $Q
done
echo "$Q: $(grep -c . $Q) jobs"
