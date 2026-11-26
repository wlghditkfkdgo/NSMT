#!/usr/bin/env bash
# MODI 2026-11-27 ((나) follow-up): same-lr comparison + order / memory controls on Sleep-EDF (32 runs), then summarize_sleep2.py.
# Run from NSMT/ in tmux:  tmux new-session -d -s flif_sleep2 "bash nsmt_flif_v1/classification/scripts/sleep2_driver.sh 0,1,2,3,0,1,2,3"
# Every line goes to classification/results/flif-v1-sleep2-pool.log (local).
set -u
GPUS=${1:-0,1,2,3,0,1,2,3}; PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -f scripts/gpu_pool2.sh ] || { echo "run from NSMT/"; exit 1; }
C=$PWD/nsmt_flif_v1/classification; PL=$C/results/flif-v1-sleep2-pool.log; R=nsmt_flif_v1/classification/scripts/run_sleep.sh
Q=$C/scripts/queues/sleep2-$(date +%Y%m%d).txt; S=flif-v1-sleep2
say() { echo "[sleep2 $(date '+%F %T')] $*" | tee -a "$PL"; }
[ -e "$C/results/$S/raw.txt" ] && { say "STOP: results/$S/raw.txt exists"; exit 1; }
mkdir -p "$(dirname "$Q")"; : > "$Q"
for f in 0 1 2 3 4; do for c in q_pop_lif_cshuf q_pop_flif_cshuf q_pop_lif_nml q_pop_flif_nml; do echo "SUITE=$S LR=0.003 bash $R $c $f" >> "$Q"; done; done
for f in 1 2 3 4; do echo "SUITE=$S LR=0.003 bash $R spkall_lif $f" >> "$Q"; for c in q_pop_lif q_pop_flif; do echo "SUITE=$S LR=0.001 bash $R $c $f" >> "$Q"; done; done
N=$(grep -c . "$Q"); say "start: $N jobs, sha $(git rev-parse --short HEAD)"
bash scripts/gpu_pool2.sh "$Q" "$GPUS" 2>&1 | tee -a "$PL"
r=$(grep -c '^RESULT' "$C/results/$S/raw.txt" 2>/dev/null); k=$(grep -c '^SKIP' "$C/results/$S/raw.txt" 2>/dev/null)
say "RESULT ${r:-0} / $N, SKIP ${k:-0}"
if [ "${r:-0}" -ne "$N" ] || [ "${k:-0}" -ne 0 ]; then say "STOP: incomplete, no summary"; exit 1; fi
(cd "$C" && "$PY" summarize_sleep2.py) 2>&1 | tee -a "$PL"
say "ALL DONE"
