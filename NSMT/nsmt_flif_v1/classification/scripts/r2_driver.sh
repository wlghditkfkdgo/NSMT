#!/usr/bin/env bash
# MODI 2026-11-23: driver of hyperparameter range 2 (lr x batch size x warmup; docs/MODI_PROJECT_LOG.md 2026-11-23, user decision 11-23
# "기준과 (나)를 둘 다 hyperparameter 조정에 넣고 결정하자"). Run from NSMT/ in tmux:
#   tmux new-session -d -s flif_r2 "bash nsmt_flif_v1/classification/scripts/r2_driver.sh 0,1,2,3,0,1,2,3"     (two runs per GPU)
#   1  select_r2.py --queue_search  -> pool (430 runs, seed 42: ETT 340 in flif-v1-ett-r2s, MIT-BIH 90 with VAL=1 in flif-v1-mitbih-r2s)
#      stop unless the two files hold 340 / 90 RESULT lines and no SKIP
#   2  select_r2.py --select        -> results/flif-v1-r2-selected.json + the final queue (validation only; fail-closed)
#   3  pool on the final queue (seeds 42 / 7 / 2026 of every model / dataset whose selected setting is not the current one;
#      flif-v1-ett-r2f, flif-v1-mitbih-r2f)
# Every line goes to classification/results/flif-v1-r2-pool.log (local). gpu_pool2.sh always prints rc=0: the SKIP lines of the raw files
# are the failure signal. Energy recount (energy_recount_ett.py --suite flif-v1-ett-r2f), summarize_r2.py and the log entry are done by hand.
set -u
GPUS=${1:-0,1,2,3,0,1,2,3}; PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -f scripts/gpu_pool2.sh ] || { echo "run from NSMT/"; exit 1; }
C=$PWD/nsmt_flif_v1/classification; FR=$PWD/nsmt_flif_v1/forecasting/results; QD=$C/scripts/queues; mkdir -p "$QD"
PL=$C/results/flif-v1-r2-pool.log; Q1=$QD/r2-search-20261123.txt; Q2=$QD/r2-final-20261123.txt
say() { echo "[driver $(date '+%F %T')] $*" | tee -a "$PL"; }
pool() { local q=$1; say "pool start: $(basename "$q") $(grep -c . "$q") jobs"; [ -s "$q" ] && bash scripts/gpu_pool2.sh "$q" "$GPUS" 2>&1 | tee -a "$PL"; say "pool done: $(basename "$q")"; }
if [ -e "$C/results/flif-v1-r2-selected.json" ] || [ -e "$FR/flif-v1-ett-r2s/raw.txt" ] || [ -e "$C/results/flif-v1-mitbih-r2s/raw.txt" ]; then
  say "STOP: range-2 results already exist (this driver starts from nothing)"; exit 1; fi
(cd "$C" && $PY select_r2.py --queue_search "$Q1") 2>&1 | tee -a "$PL"
if [ "${PIPESTATUS[0]}" -ne 0 ] || [ ! -s "$Q1" ]; then say "STOP: no search queue"; exit 1; fi
pool "$Q1"
e=$(grep -c '^RESULT' "$FR/flif-v1-ett-r2s/raw.txt" 2>/dev/null); m=$(grep -c '^RESULT' "$C/results/flif-v1-mitbih-r2s/raw.txt" 2>/dev/null)
k=$(cat "$FR/flif-v1-ett-r2s/raw.txt" "$C/results/flif-v1-mitbih-r2s/raw.txt" 2>/dev/null | grep -c '^SKIP')
say "search: ETT ${e:-0} RESULT, MIT-BIH ${m:-0} RESULT, ${k:-0} SKIP (expected 340, 90, 0)"
if [ "${e:-0}" -ne 340 ] || [ "${m:-0}" -ne 90 ] || [ "${k:-0}" -ne 0 ]; then say "STOP: search incomplete"; exit 1; fi
rm -f "$Q2"
(cd "$C" && $PY select_r2.py --select --queue_final "$Q2") 2>&1 | tee -a "$PL"
if [ "${PIPESTATUS[0]}" -ne 0 ] || [ ! -e "$Q2" ]; then say "STOP: selection failed"; exit 1; fi
pool "$Q2"
e=$(grep -c '^RESULT' "$FR/flif-v1-ett-r2f/raw.txt" 2>/dev/null); m=$(grep -c '^RESULT' "$C/results/flif-v1-mitbih-r2f/raw.txt" 2>/dev/null)
k=$(cat "$FR/flif-v1-ett-r2f/raw.txt" "$C/results/flif-v1-mitbih-r2f/raw.txt" 2>/dev/null | grep -c '^SKIP')
say "final: ETT ${e:-0} RESULT, MIT-BIH ${m:-0} RESULT, ${k:-0} SKIP"
say "ALL DONE (run the energy recount of flif-v1-ett-r2f, then summarize_r2.py)"
