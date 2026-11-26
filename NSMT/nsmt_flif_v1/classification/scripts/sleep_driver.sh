#!/usr/bin/env bash
# MODI 2026-11-26 ((나)): Sleep-EDF SNN comparison, stage 1 (lr on fold-0 validation, 10 runs) -> select_sleep.py -> stage 2 (folds 1-4,
# 20 runs) -> summarize_sleep.py. Run from NSMT/ in tmux:
#   tmux new-session -d -s flif_sleep "bash nsmt_flif_v1/classification/scripts/sleep_driver.sh 0,1,2,3,0,1,2,3"
# Every line goes to classification/results/flif-v1-sleep-pool.log (local).
set -u
GPUS=${1:-0,1,2,3,0,1,2,3}; PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -f scripts/gpu_pool2.sh ] || { echo "run from NSMT/"; exit 1; }
C=$PWD/nsmt_flif_v1/classification; PL=$C/results/flif-v1-sleep-pool.log; R=nsmt_flif_v1/classification/scripts/run_sleep.sh
Q1=$C/scripts/queues/sleep-lr-$(date +%Y%m%d).txt; Q2=$C/scripts/queues/sleep-main-$(date +%Y%m%d).txt
say() { echo "[sleep $(date '+%F %T')] $*" | tee -a "$PL"; }
for s in flif-v1-sleep-lr flif-v1-sleep; do [ -e "$C/results/$s/raw.txt" ] && { say "STOP: results/$s/raw.txt exists"; exit 1; }; done
mkdir -p "$(dirname "$Q1")"; : > "$Q1"
for lr in 0.001 0.003; do for c in spkall_lif q_pop_lif q_pop_flif kv_pop_lif kv_pop_flif; do echo "SUITE=flif-v1-sleep-lr LR=$lr bash $R $c 0" >> "$Q1"; done; done
say "stage 1 start: $(grep -c . "$Q1") jobs, sha $(git rev-parse --short HEAD)"
bash scripts/gpu_pool2.sh "$Q1" "$GPUS" 2>&1 | tee -a "$PL"
r=$(grep -c '^RESULT' "$C/results/flif-v1-sleep-lr/raw.txt" 2>/dev/null); k=$(grep -c '^SKIP' "$C/results/flif-v1-sleep-lr/raw.txt" 2>/dev/null)
say "stage 1: RESULT ${r:-0} / 10, SKIP ${k:-0}"
if [ "${r:-0}" -ne 10 ] || [ "${k:-0}" -ne 0 ]; then say "STOP: stage 1 incomplete"; exit 1; fi
(cd "$C" && "$PY" select_sleep.py --queue "$Q2") 2>&1 | tee -a "$PL"
[ -s "$Q2" ] && [ "$(grep -c . "$Q2")" -eq 20 ] || { say "STOP: no stage-2 queue"; exit 1; }
say "stage 2 start: 20 jobs"
bash scripts/gpu_pool2.sh "$Q2" "$GPUS" 2>&1 | tee -a "$PL"
r=$(grep -c '^RESULT' "$C/results/flif-v1-sleep/raw.txt" 2>/dev/null); k=$(grep -c '^SKIP' "$C/results/flif-v1-sleep/raw.txt" 2>/dev/null)
say "stage 2: RESULT ${r:-0} / 20, SKIP ${k:-0}"
if [ "${r:-0}" -ne 20 ] || [ "${k:-0}" -ne 0 ]; then say "STOP: stage 2 incomplete, no summary"; exit 1; fi
(cd "$C" && "$PY" summarize_sleep.py) 2>&1 | tee -a "$PL"
say "ALL DONE"
