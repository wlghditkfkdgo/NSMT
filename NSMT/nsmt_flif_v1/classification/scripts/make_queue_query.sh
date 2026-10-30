#!/usr/bin/env bash
# MODI 2026-10-30: regenerates the (git-ignored) queues of the query batch (docs/MODI_PROJECT_LOG.md 2026-10-30) -- the dual-path
# model with the neocortex as the attention query (nsmt4q_<n> + nsmt4q_lif_nomem), B-form settings, neocortex-only neuron swap.
#   smoke: every condition once, 1 epoch, seed 42 (CinC fold 0; ETT ETTh2 H96)
#   main : CinC fold 0, 90 epochs (15 runs); ETT ETTh2 x H96 / H720 (user scope for quick checks, 10-30; 30 runs); seeds 42 / 7 / 2026
# Pools (from NSMT/): nsmt_flif_v1/classification/scripts/query_tune_driver.sh
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"
CIN=nsmt_flif_v1/classification/scripts/run_rhythm_addneo.sh; ETT=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh
NQ="$(for n in pop_flif pop_lif flif lif; do echo -n "nsmt4q_$n "; done)nsmt4q_lif_nomem"
Q=$QD/query-smoke-20261030.txt; : > $Q
for c in $NQ; do echo "COND=$c DATA=CINC2017 SUITE=smoke-query-cinc-20261030 LR=0.001 EPOCHS=1 bash $CIN 42"; done >> $Q
for c in $NQ; do echo "COND=$c SUITE=smoke-query-ett-20261030 EPOCHS=1 bash $ETT ETTh2 96 42"; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
Q=$QD/query-cinc-20261030.txt; : > $Q
for c in $NQ; do for s in $SEEDS; do echo "COND=$c DATA=CINC2017 SUITE=flif-v1-rhythm-query-cinc LR=0.001 EPOCHS=90 bash $CIN $s"; done; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
Q=$QD/query-ett-20261030.txt; : > $Q
for pl in 96 720; do for c in $NQ; do for s in $SEEDS; do echo "COND=$c SUITE=flif-v1-ett-query bash $ETT ETTh2 $pl $s"; done; done; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
