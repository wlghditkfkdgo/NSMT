#!/usr/bin/env bash
# MODI 2026-11-27 (user decisions 11-27 "1. 진행", "2. 축적만 우선 진행"): Neocortex membrane readout and past-reconstruction aux on
# Sleep-EDF (60 runs, lr 0.003), then summarize_sleep3.py. Run from NSMT/ in tmux:
#   tmux new-session -d -s flif_sleep3 "bash nsmt_flif_v1/classification/scripts/sleep3_driver.sh 0,1,2,3,0,1,2,3"
# Every line goes to classification/results/flif-v1-sleep3-pool.log (local).
set -u
GPUS=${1:-0,1,2,3,0,1,2,3}; PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -f scripts/gpu_pool2.sh ] || { echo "run from NSMT/"; exit 1; }
C=$PWD/nsmt_flif_v1/classification; PL=$C/results/flif-v1-sleep3-pool.log; R=nsmt_flif_v1/classification/scripts/run_sleep.sh
Q=$C/scripts/queues/sleep3-$(date +%Y%m%d).txt; S=flif-v1-sleep3
say() { echo "[sleep3 $(date '+%F %T')] $*" | tee -a "$PL"; }
[ -e "$C/results/$S/raw.txt" ] && { say "STOP: results/$S/raw.txt exists"; exit 1; }
mkdir -p "$(dirname "$Q")"; : > "$Q"
for f in 0 1 2 3 4; do for n in pop_lif pop_flif; do for v in mem_acc mem acc mem_cshuf mem_nml acc_cshuf; do
  echo "SUITE=$S LR=0.003 bash $R q_${n}_$v $f" >> "$Q"; done; done; done
N=$(grep -c . "$Q"); say "start: $N jobs, sha $(git rev-parse --short HEAD)"
bash scripts/gpu_pool2.sh "$Q" "$GPUS" 2>&1 | tee -a "$PL"
r=$(grep -c '^RESULT' "$C/results/$S/raw.txt" 2>/dev/null); k=$(grep -c '^SKIP' "$C/results/$S/raw.txt" 2>/dev/null)
say "RESULT ${r:-0} / $N, SKIP ${k:-0}"
if [ "${r:-0}" -ne "$N" ] || [ "${k:-0}" -ne 0 ]; then say "STOP: incomplete, no summary"; exit 1; fi
(cd "$C" && "$PY" summarize_sleep3.py) 2>&1 | tee -a "$PL"
say "ALL DONE"
