#!/usr/bin/env bash
# MODI 2026-11-18: regenerates the (git-ignored) phase-A queue of the long-input learning-rate test (docs/MODI_PROJECT_LOG.md; step 2
# of user decision 11-18 "1. 진행  2. 1번에서 개선이 없으면 진행": training settings re-chosen for long inputs, because at input 720
# 54 / 84 runs had their best validation at the first epoch).
#   suite flif-v1-ett-lr-repro (3 runs, report only): spk_ssa / nsmt4q_pop_<lif|flif>_msn, ETTh2 H96 seed 42, input 720, default lr,
#         retrained at this sha (the default arm is reused from the input-length suites flif-v1-ett-len336 / -len720)
#   suites flif-v1-ett-lr<L>-<LR> (72 runs): the three compared models x ETTh2 / ETTm2 x H96 / H720 x input L in {336, 720} x
#         LR in {5e-05, 0.0001, 0.00025} (the default is 0.0005 for both datasets), seed 42 only -- the selection phase.
#   Phase B (seeds 7 / 2026 at each model's selected lr) is written by select_lrlong.py after phase A.
# LR is written as Python prints the float (the runner finds the run folder by '+lr+<LR>+').
# Order: the retrain runs first, then input 720 before 336, ETTm2 before ETTh2, the lowest lr first (longest runs first).
# NSMT/: bash scripts/gpu_pool2.sh <queue> <gpus> (appends the gpu); normally started by scripts/lrlong_driver.sh.
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
ET=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh; MX=nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh
Q=$QD/lrlong-a-20261118.txt; : > $Q
R=flif-v1-ett-lr-repro
echo "COND=spk_ssa SUITE=$R SEQ=720 bash $MX ETTh2 96 42" >> $Q
for n in pop_lif pop_flif; do echo "COND=nsmt4q_${n}_msn SUITE=$R SEQ=720 bash $ET ETTh2 96 42" >> $Q; done
for L in 720 336; do for ds in ETTm2 ETTh2; do for pl in 720 96; do for lr in 5e-05 0.0001 0.00025; do
  SU=flif-v1-ett-lr$L-$lr
  echo "COND=spk_ssa SUITE=$SU SEQ=$L LR=$lr bash $MX $ds $pl 42"
  for n in pop_lif pop_flif; do echo "COND=nsmt4q_${n}_msn SUITE=$SU SEQ=$L LR=$lr bash $ET $ds $pl 42"; done
done; done; done; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
