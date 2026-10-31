#!/usr/bin/env bash
# MODI 2026-10-31: regenerates the (git-ignored) queues of the CinC follow-up (docs/MODI_PROJECT_LOG.md 2026-10-31; user decisions):
#   main: (B) nsmt4q_pop_{lif,flif}_nomem fold 0 (6) and (C) mixall_lif_bidir / mixall_lif_bidir_rzero fold 0 (6) -- suite
#         flif-v1-rhythm-followup-cinc; (A) nsmt4q_pop_{lif,flif} folds 1-4 (24) -- suite flif-v1-rhythm-query-cv-cinc;
#         CinC, 90 epochs, lr 0.001, seeds 42 / 7 / 2026; short runs first
#   smoke: each new condition once + one fold-1 query run, 1 epoch, seed 42
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"; CIN=nsmt_flif_v1/classification/scripts/run_rhythm_addneo.sh
Q=$QD/followup-smoke-20261031.txt; : > $Q
for c in nsmt4q_pop_lif_nomem nsmt4q_pop_flif_nomem mixall_lif_bidir mixall_lif_bidir_rzero; do
  echo "COND=$c DATA=CINC2017 SUITE=smoke-followup-cinc-20261031 LR=0.001 EPOCHS=1 bash $CIN 42"; done >> $Q
echo "COND=nsmt4q_pop_lif FOLD=1 DATA=CINC2017 SUITE=smoke-followup-cinc-20261031 LR=0.001 EPOCHS=1 bash $CIN 42" >> $Q
echo "$Q: $(grep -c . $Q) jobs"
Q=$QD/followup-20261031.txt; : > $Q
for c in nsmt4q_pop_lif_nomem nsmt4q_pop_flif_nomem mixall_lif_bidir mixall_lif_bidir_rzero; do for s in $SEEDS; do
  echo "COND=$c DATA=CINC2017 SUITE=flif-v1-rhythm-followup-cinc LR=0.001 EPOCHS=90 bash $CIN $s"; done; done >> $Q
for f in 1 2 3 4; do for c in nsmt4q_pop_flif nsmt4q_pop_lif; do for s in $SEEDS; do
  echo "COND=$c FOLD=$f DATA=CINC2017 SUITE=flif-v1-rhythm-query-cv-cinc LR=0.001 EPOCHS=90 bash $CIN $s"; done; done; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
