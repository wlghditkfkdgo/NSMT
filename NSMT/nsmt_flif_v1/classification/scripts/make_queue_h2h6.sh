#!/usr/bin/env bash
# MODI 2026-11-04: regenerates the (git-ignored) queue of the head test H2 / H6 (docs/MODI_PROJECT_LOG.md 2026-11-04; user decision
# "H2와 H6을 간단하게 검증해보자"; new heads on our models only, user decision 3 of 11-03):
#   H2: nsmt4q_pop_{lif,flif}_hattn x seeds 42 / 7 / 2026 on MIT-BIH, default settings (lr 0.01, full training file, last epoch tested)
#       -- suite flif-v1-mitbih-h2 (6 runs; baseline = flif-v1-mitbih-query)
#   H6: nsmt4q_pop_{lif,flif}_h6 x ETTh2 / ETTm2 x H96 / H720 x seeds, unchanged run_ett_dual.sh settings
#       -- suite flif-v1-ett-h6 (24 runs; baseline = flif-v1-ett-query (ETTh2) + flif-v1-ett-query-m2 (ETTm2))
# Order: the long ETTm2 runs first, then MIT-BIH, then ETTh2. NSMT/: bash scripts/gpu_pool2.sh <queue> <gpus> (appends the gpu).
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"; MB=nsmt_flif_v1/classification/scripts/run_mitbih4.sh; ET=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh
Q=$QD/head-h2h6-20261104.txt; : > $Q
for pl in 720 96; do for c in nsmt4q_pop_lif_h6 nsmt4q_pop_flif_h6; do for s in $SEEDS; do
  echo "COND=$c SUITE=flif-v1-ett-h6 bash $ET ETTm2 $pl $s"; done; done; done >> $Q
for c in nsmt4q_pop_lif_hattn nsmt4q_pop_flif_hattn; do for s in $SEEDS; do
  echo "COND=$c SUITE=flif-v1-mitbih-h2 bash $MB $s"; done; done >> $Q
for pl in 96 720; do for c in nsmt4q_pop_lif_h6 nsmt4q_pop_flif_h6; do for s in $SEEDS; do
  echo "COND=$c SUITE=flif-v1-ett-h6 bash $ET ETTh2 $pl $s"; done; done; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
