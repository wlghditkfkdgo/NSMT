#!/usr/bin/env bash
# MODI 2026-11-27: runs the PTB order check, then its pre-registered summary. Run from NSMT/ in tmux:
#   tmux new-session -d -s flif_oc3 "bash nsmt_flif_v1/classification/scripts/oc3_driver.sh 0,1,2,3,0,1,2,3"
# Every line goes to classification/results/flif-v1-oc3-pool.log (local).
set -u
GPUS=${1:-0,1,2,3,0,1,2,3}; PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -f scripts/gpu_pool2.sh ] || { echo "run from NSMT/"; exit 1; }
C=$PWD/nsmt_flif_v1/classification; SUITE=flif-v1-oc3; PL=$C/results/flif-v1-oc3-pool.log
Q=$C/scripts/queues/oc3-$(date +%Y%m%d).txt
say() { echo "[oc3 $(date '+%F %T')] $*" | tee -a "$PL"; }
[ -e "$C/results/$SUITE/raw.txt" ] && { say "STOP: results/$SUITE/raw.txt exists"; exit 1; }
bash "$C/scripts/make_queue_oc3.sh" "$Q" "$SUITE" | tee -a "$PL"
N=$(grep -c . "$Q")
say "pool start: $(basename "$Q") $N jobs, sha $(git rev-parse --short HEAD)"
bash scripts/gpu_pool2.sh "$Q" "$GPUS" 2>&1 | tee -a "$PL"
say "pool done"
r=$(grep -c '^RESULT' "$C/results/$SUITE/raw.txt" 2>/dev/null); k=$(grep -c '^SKIP' "$C/results/$SUITE/raw.txt" 2>/dev/null)
say "RESULT ${r:-0} / $N, SKIP ${k:-0}"
if [ "${r:-0}" -ne "$N" ] || [ "${k:-0}" -ne 0 ]; then say "STOP: incomplete, no summary"; exit 1; fi
(cd "$C" && "$PY" summarize_oc3.py --suite "$SUITE") 2>&1 | tee -a "$PL"
say "ALL DONE"
