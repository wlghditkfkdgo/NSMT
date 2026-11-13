#!/usr/bin/env bash
# MODI 2026-11-13: regenerates the (git-ignored) queue of the STFT test (docs/MODI_PROJECT_LOG.md; user request 11-13: STFT input of
# the Neocortex, "(가) + soma 없는 pop과 (나)를 동시에 구현 및 실험하고 비교"; base = Neocortex MS):
#   nsmt4q_pop_<lif|flif>_msn_<stft|fire_stft|fire_same> x seeds 42 / 7 / 2026
#     ETT: ETTh2 / ETTm2 x H96 / H720, unchanged run_ett_dual.sh settings -- suite flif-v1-ett-stft (72 runs; baseline = the msn runs
#          of flif-v1-ett-ms)
#     MIT-BIH: unchanged run_mitbih4.sh defaults (lr 0.01, full training file, last epoch tested) -- suite flif-v1-mitbih-stft (18 runs;
#          baseline = the msn runs of flif-v1-mitbih-ms)
# Order: every pop LIF run first, then pop f-LIF; inside each, ETTm2 H720, ETTm2 H96, MIT-BIH, ETTh2.
# NSMT/: bash scripts/gpu_pool2.sh <queue> <gpus> (appends the gpu).
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"; MB=nsmt_flif_v1/classification/scripts/run_mitbih4.sh; ET=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh
Q=$QD/stft-20261113.txt; : > $Q
for n in pop_lif pop_flif; do
  for pl in 720 96; do for v in stft fire_stft fire_same; do for s in $SEEDS; do
    echo "COND=nsmt4q_${n}_msn_${v} SUITE=flif-v1-ett-stft bash $ET ETTm2 $pl $s"; done; done; done >> $Q
  for v in stft fire_stft fire_same; do for s in $SEEDS; do
    echo "COND=nsmt4q_${n}_msn_${v} SUITE=flif-v1-mitbih-stft bash $MB $s"; done; done >> $Q
  for pl in 96 720; do for v in stft fire_stft fire_same; do for s in $SEEDS; do
    echo "COND=nsmt4q_${n}_msn_${v} SUITE=flif-v1-ett-stft bash $ET ETTh2 $pl $s"; done; done; done >> $Q
done
echo "$Q: $(grep -c . $Q) jobs"
