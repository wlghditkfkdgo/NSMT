#!/usr/bin/env bash
# MODI 2026-11-20: regenerates the (git-ignored) queue of the aux-target memory check (docs/MODI_PROJECT_LOG.md 2026-11-20; user decision
# 11-20 "4-1 검증(작음) -> 범위 1 -> 4-2 (가) + (나)"). 72 runs, seed 42, default PE=0 (no relative position bias):
#   the all-SpikeDE base with the raw-patch aux, Neocortex variants memory / no memory / shuffled order
#   (nsmt4q_pop_<lif|flif>_msn_sde{,_nomem,_nshuf}_auxf), aux target shift s in {0, 2, 4, -2} (env AUXS; unset = 0)
#   suites flif-v1-auxck-ett-<s0|s2|s4|sm2> (ETTh2 / ETTm2 H96, run_ett_dual.sh) and flif-v1-auxck-mb-<...> (MIT-BIH, VAL=1, run_mitbih4.sh)
# Order: MIT-BIH (longest), ETTm2, ETTh2. NSMT/: bash scripts/gpu_pool2.sh <queue> <gpus> (appends the gpu).
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
ET=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh; M4=nsmt_flif_v1/classification/scripts/run_mitbih4.sh
Q=$QD/auxck-20261120.txt; : > $Q
TG="s0: s2:2 s4:4 sm2:-2"
for t in $TG; do tn=${t%%:*}; s=${t#*:}; for n in lif flif; do for v in "" _nomem _nshuf; do
  echo "COND=nsmt4q_pop_${n}_msn_sde${v}_auxf SUITE=flif-v1-auxck-mb-$tn ${s:+AUXS=$s }VAL=1 bash $M4 42"; done; done; done >> $Q
for ds in ETTm2 ETTh2; do for t in $TG; do tn=${t%%:*}; s=${t#*:}; for n in lif flif; do for v in "" _nomem _nshuf; do
  echo "COND=nsmt4q_pop_${n}_msn_sde${v}_auxf SUITE=flif-v1-auxck-ett-$tn ${s:+AUXS=$s }bash $ET $ds 96 42"; done; done; done; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
