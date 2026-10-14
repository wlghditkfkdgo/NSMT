#!/usr/bin/env bash
# nsmt_flif_v1 smoke: 1 epoch of every condition on ETTh1 H96 seed 42 (pipeline check, NOT an experiment).
# usage: smoke.sh <gpu>   env: SUITE (smoke-flif-<date>), CONDS
set -u
GPU=${1:?}; W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SUITE=${SUITE:-smoke-flif-$(date +%Y%m%d)}; export EPOCHS=${EPOCHS:-1}
for c in ${CONDS:-base neo_lif_both neo_lif_mlp neo_flif_both neo_flif_mlp hippo_only}; do
  t0=$(date +%s); COND=$c bash "$W/scripts/run_ett.sh" ETTh1 96 42 "$GPU" > /dev/null; rc=$?
  line=$(tail -1 "$W/results/$SUITE/raw.txt")
  echo "SMOKE cond=$c rc=$rc time_s=$(( $(date +%s) - t0 )) :: $line"
done
