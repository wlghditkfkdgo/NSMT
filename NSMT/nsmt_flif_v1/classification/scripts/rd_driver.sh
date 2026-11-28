#!/usr/bin/env bash
# MODI 2026-11-28 (approved plan /home/yschoi/.claude/plans/1-snuggly-galaxy.md; user decisions 11-27 "1. 둘다", "2. 일단은 빼고 성능
# 변화만"): Neocortex readout (spk / mem / memd) x accumulation aux x LIF / f-LIF / pop LIF on three tasks, in three phases:
#   M  MIT-BIH  51 runs  (rd_<n>_<r>[_acc] x seeds 42 / 7 / 2026; suite flif-v1-mitbih-rd)      -> summarize_rd.py --task mitbih
#   S  Sleep    40 runs  (8 fill-in models x folds 0-4, lr 0.003; suite flif-v1-sleep5)          -> summarize_rd.py --task sleep
#   E  ETT     204 runs  (17 models x ETTm2 / ETTh2 x H96 / H720 x 3 seeds; suite flif-v1-ett-rd) -> summarize_rd.py --task ett, --task all
# A phase that ends with a RESULT count other than its job count, or with a SKIP line, stops the driver (later phases do not run).
# Run from NSMT/ in tmux:
#   tmux new-session -d -s flif_rd "bash nsmt_flif_v1/classification/scripts/rd_driver.sh 0,1,2,3,0,1,2,3"
# Every line goes to classification/results/flif-v1-rd-pool.log (local). No commits and no code edits while it runs.
set -u
GPUS=${1:-0,1,2,3,0,1,2,3}; PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -f scripts/gpu_pool2.sh ] || { echo "run from NSMT/"; exit 1; }
C=$PWD/nsmt_flif_v1/classification; F=$PWD/nsmt_flif_v1/forecasting; PL=$C/results/flif-v1-rd-pool.log
RM=nsmt_flif_v1/classification/scripts/run_mitbih4.sh; RS=nsmt_flif_v1/classification/scripts/run_sleep.sh
RE=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh; QD=$C/scripts/queues; TS=$(date +%Y%m%d)
say() { echo "[rd $(date '+%F %T')] $*" | tee -a "$PL"; }
mkdir -p "$QD"
MODELS=""; for n in lif flif pop_lif; do for r in spk mem memd; do for a in "" _acc; do
  [ "$n$r$a" = pop_lifspk ] || MODELS="$MODELS rd_${n}_${r}$a"; done; done; done        # pop_lif spk off = the r1 reference runs
FILL="q_lif_memd q_lif_memd_acc q_flif_memd q_flif_memd_acc q_flif_acc q_flif_mem_acc q_pop_lif_memd q_pop_lif_memd_acc"

phase() {   # phase <name> <queue> <raw.txt> <summary task>
  local N r k; N=$(grep -c . "$2")
  say "phase $1 start: $N jobs, sha $(git rev-parse --short HEAD)"
  bash scripts/gpu_pool2.sh "$2" "$GPUS" 2>&1 | tee -a "$PL"
  r=$(grep -c '^RESULT' "$3" 2>/dev/null); k=$(grep -c '^SKIP' "$3" 2>/dev/null)
  say "phase $1: RESULT ${r:-0} / $N, SKIP ${k:-0}"
  if [ "${r:-0}" -ne "$N" ] || [ "${k:-0}" -ne 0 ]; then say "STOP: phase $1 incomplete, no summary, later phases not run"; exit 1; fi
  (cd "$C" && "$PY" summarize_rd.py --task "$4") 2>&1 | tee -a "$PL"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { say "STOP: summarize_rd.py --task $4 failed"; exit 1; }
}

for S in flif-v1-mitbih-rd flif-v1-sleep5; do [ -e "$C/results/$S/raw.txt" ] && { say "STOP: results/$S/raw.txt exists"; exit 1; }; done
[ -e "$F/results/flif-v1-ett-rd/raw.txt" ] && { say "STOP: forecasting results/flif-v1-ett-rd/raw.txt exists"; exit 1; }

QM=$QD/rd-mitbih-$TS.txt; : > "$QM"
for s in 42 7 2026; do for c in $MODELS; do echo "COND=$c SUITE=flif-v1-mitbih-rd bash $RM $s" >> "$QM"; done; done
phase M "$QM" "$C/results/flif-v1-mitbih-rd/raw.txt" mitbih

QS=$QD/rd-sleep-$TS.txt; : > "$QS"
for f in 0 1 2 3 4; do for c in $FILL; do echo "SUITE=flif-v1-sleep5 LR=0.003 bash $RS $c $f" >> "$QS"; done; done
phase S "$QS" "$C/results/flif-v1-sleep5/raw.txt" sleep

QE=$QD/rd-ett-$TS.txt; : > "$QE"
for cell in "ETTm2 96" "ETTm2 720" "ETTh2 96" "ETTh2 720"; do for s in 42 7 2026; do for c in $MODELS; do
  echo "COND=$c SUITE=flif-v1-ett-rd bash $RE $cell $s" >> "$QE"; done; done; done
phase E "$QE" "$F/results/flif-v1-ett-rd/raw.txt" ett

(cd "$C" && "$PY" summarize_rd.py --task all) 2>&1 | tee -a "$PL"
say "ALL DONE"
