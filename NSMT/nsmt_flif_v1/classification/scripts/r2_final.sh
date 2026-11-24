#!/usr/bin/env bash
# MODI 2026-11-24: the final stage of hyperparameter range 2, run by hand. The search driver (r2_driver.sh) stopped at "search incomplete":
# 5 finished ETT search runs had a SKIP line because the runners' folder lookup had no lr when LR was unset (fixed; check_r2.py f). Their
# RESULT lines were rebuilt by recover_r2_rows.py, and select_r2.py --select made the selection with the pre-registered rules unchanged
# (results/flif-v1-r2-selected.json, queue scripts/queues/r2-final-20261123.txt; docs/MODI_PROJECT_LOG.md 2026-11-24). This is step 3 of
# r2_driver.sh on that queue. Run from NSMT/ in tmux:
#   tmux new-session -d -s flif_r2f "bash nsmt_flif_v1/classification/scripts/r2_final.sh 0,1,2,3,0,1,2,3"     (two runs per GPU)
# Every line goes to classification/results/flif-v1-r2-pool.log (local). Energy recount, summarize_r2.py and the log entry are done by hand.
set -u
GPUS=${1:-0,1,2,3,0,1,2,3}; PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -f scripts/gpu_pool2.sh ] || { echo "run from NSMT/"; exit 1; }
C=$PWD/nsmt_flif_v1/classification; FR=$PWD/nsmt_flif_v1/forecasting/results
PL=$C/results/flif-v1-r2-pool.log; Q2=$C/scripts/queues/r2-final-20261123.txt
say() { echo "[final $(date '+%F %T')] $*" | tee -a "$PL"; }
[ -e "$C/results/flif-v1-r2-selected.json" ] && [ -s "$Q2" ] || { say "STOP: no selection file or final queue"; exit 1; }
if [ -e "$FR/flif-v1-ett-r2f/raw.txt" ] || [ -e "$C/results/flif-v1-mitbih-r2f/raw.txt" ]; then say "STOP: final results already exist"; exit 1; fi
ne=$(grep -c 'run_ett_' "$Q2"); nm=$(grep -c 'run_mitbih4' "$Q2")
say "pool start: $(basename "$Q2") $(grep -c . "$Q2") jobs (ETT $ne, MIT-BIH $nm), sha $(git rev-parse --short HEAD)"
bash scripts/gpu_pool2.sh "$Q2" "$GPUS" 2>&1 | tee -a "$PL"
say "pool done: $(basename "$Q2")"
e=$(grep -c '^RESULT' "$FR/flif-v1-ett-r2f/raw.txt" 2>/dev/null); m=$(grep -c '^RESULT' "$C/results/flif-v1-mitbih-r2f/raw.txt" 2>/dev/null)
k=$(cat "$FR/flif-v1-ett-r2f/raw.txt" "$C/results/flif-v1-mitbih-r2f/raw.txt" 2>/dev/null | grep -c '^SKIP')
say "final: ETT ${e:-0} RESULT, MIT-BIH ${m:-0} RESULT, ${k:-0} SKIP (expected $ne, $nm, 0)"
if [ "${e:-0}" -ne "$ne" ] || [ "${m:-0}" -ne "$nm" ] || [ "${k:-0}" -ne 0 ]; then say "STOP: final incomplete"; exit 1; fi
say "ALL DONE (run the energy recount of flif-v1-ett-r2f, then summarize_r2.py)"
