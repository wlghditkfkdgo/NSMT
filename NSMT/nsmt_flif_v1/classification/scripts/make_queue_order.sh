#!/usr/bin/env bash
# MODI 2026-11-02: regenerates the (git-ignored) queues of the memory / order batch (docs/MODI_PROJECT_LOG.md 2026-11-02; user decision):
#   main: nsmt4q_pop_{lif,flif}_nshuf (Neocortex-only order control) folds 0-4 (30 runs, fold 0 first) and nsmt4q_pop_{lif,flif}_nomem
#         folds 1-4 (24 runs) -- suite flif-v1-rhythm-order-cinc; CinC, 90 epochs, lr 0.001, seeds 42 / 7 / 2026; long runs first
#   smoke: each new condition once (1 epoch, seed 42), one of them on fold 2
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"; CIN=nsmt_flif_v1/classification/scripts/run_rhythm_addneo.sh
Q=$QD/order-smoke-20261102.txt; : > $Q
echo "COND=nsmt4q_pop_lif_nshuf DATA=CINC2017 SUITE=smoke-order-cinc-20261102 LR=0.001 EPOCHS=1 bash $CIN 42" >> $Q
echo "COND=nsmt4q_pop_flif_nshuf FOLD=2 DATA=CINC2017 SUITE=smoke-order-cinc-20261102 LR=0.001 EPOCHS=1 bash $CIN 42" >> $Q
echo "COND=nsmt4q_pop_lif_nomem FOLD=1 DATA=CINC2017 SUITE=smoke-order-cinc-20261102 LR=0.001 EPOCHS=1 bash $CIN 42" >> $Q
echo "$Q: $(grep -c . $Q) jobs"
Q=$QD/order-20261102.txt; : > $Q
for f in 0 1 2 3 4; do for c in nsmt4q_pop_flif_nshuf nsmt4q_pop_lif_nshuf; do for s in $SEEDS; do
  echo "COND=$c FOLD=$f DATA=CINC2017 SUITE=flif-v1-rhythm-order-cinc LR=0.001 EPOCHS=90 bash $CIN $s"; done; done; done >> $Q
for f in 1 2 3 4; do for c in nsmt4q_pop_flif_nomem nsmt4q_pop_lif_nomem; do for s in $SEEDS; do
  echo "COND=$c FOLD=$f DATA=CINC2017 SUITE=flif-v1-rhythm-order-cinc LR=0.001 EPOCHS=90 bash $CIN $s"; done; done; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
