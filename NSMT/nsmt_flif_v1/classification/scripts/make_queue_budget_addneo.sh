#!/usr/bin/env bash
# MODI 2026-10-26: regenerates the (git-ignored) queue files of the training-budget suite (A) and the added-neocortex suite (B)
# exactly as they were run: long runs first, seeds 42 / 7 / 2026, CinC fold 0. Pools: from NSMT/,
#   bash scripts/gpu_pool2.sh <queue> 0,1,2,3,0,1,2,3   (A, 2 per GPU)      bash scripts/gpu_pool2.sh <queue> 0,1,2,3   (B, 1 per GPU)
# usage: make_queue_budget_addneo.sh [A|B|both]
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; Q=$W/scripts/queues
case ${1:-both} in A|both)
  : > $Q/rhythm-budget-20261026.txt
  for c in mix_pop_bidir mix_pop_lif_bidir mix_flif_bidir mix_lif_bidir spk_ssa; do for s in 42 7 2026; do
    echo "COND=$c DATA=CINC2017 SUITE=flif-v1-rhythm-budget-cinc LR=0.001 EPOCHS=90 bash nsmt_flif_v1/classification/scripts/run_rhythm6.sh $s"
  done; done >> $Q/rhythm-budget-20261026.txt ;; esac
case ${1:-both} in B|both)
  : > $Q/rhythm-addneo-20261026.txt
  for c in spkneo_pop_flif spkneo_pop_lif spkneo_flif spkneo_lif spkneo_lif_nomem; do for s in 42 7 2026; do
    echo "COND=$c DATA=CINC2017 SUITE=flif-v1-rhythm-addneo-cinc LR=0.001 bash nsmt_flif_v1/classification/scripts/run_rhythm_addneo.sh $s"
  done; done >> $Q/rhythm-addneo-20261026.txt ;; esac
