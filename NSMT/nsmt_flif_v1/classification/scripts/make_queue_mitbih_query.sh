#!/usr/bin/env bash
# MODI 2026-11-01: regenerates the (git-ignored) queues of the MIT-BIH query batch (docs/MODI_PROJECT_LOG.md 2026-11-01; user decision):
#   stage1: nsmt4q_pop_{lif,flif} x seeds 42 / 7 / 2026, default settings (lr 0.01, full training file) -- suite flif-v1-mitbih-query
#   tune:   only if neither query population is 'better' than Spikformer-style in stage 1 (summarize_mitbih_query.py prints the
#           trigger): peak lr {0.0025, 0.005, 0.01, 0.02} x {nsmt4q_pop_lif, nsmt4q_pop_flif, spk_ssa} x seeds, VAL=1 (80 / 20 split of
#           the training file) -- suite flif-v1-mitbih-query-tune
# usage: make_queue_mitbih_query.sh stage1|tune
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"; MB=nsmt_flif_v1/classification/scripts/run_mitbih4.sh
case ${1:?stage1|tune} in
  stage1)
    Q=$QD/mitbih-query-20261101.txt; : > $Q
    for c in nsmt4q_pop_flif nsmt4q_pop_lif; do for s in $SEEDS; do
      echo "COND=$c SUITE=flif-v1-mitbih-query bash $MB $s"; done; done >> $Q ;;
  tune)
    Q=$QD/mitbih-query-tune-20261101.txt; : > $Q
    for c in nsmt4q_pop_flif nsmt4q_pop_lif spk_ssa; do for lr in 0.0025 0.005 0.01 0.02; do for s in $SEEDS; do
      echo "COND=$c LR=$lr VAL=1 SUITE=flif-v1-mitbih-query-tune bash $MB $s"; done; done; done >> $Q ;;
  *) echo "bad stage"; exit 1 ;;
esac
echo "$Q: $(grep -c . $Q) jobs"
