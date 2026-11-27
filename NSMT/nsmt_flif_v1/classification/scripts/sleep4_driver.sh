#!/usr/bin/env bash
# MODI 2026-11-27 (user decisions 11-27: "(b) + 지금까지 시도해왔던 수정 사항들에 대해서 standard LIF neuron으로만 재실험 진행",
# "기본으로 LIF, f-LIF, pop LIF 이렇게 3종만 넣어줘"): the plain LIF / f-LIF batch on Sleep-EDF (70 runs), then summarize_sleep4.py.
# Run from NSMT/ in tmux:
#   tmux new-session -d -s flif_sleep4 "bash nsmt_flif_v1/classification/scripts/sleep4_driver.sh 0,1,2,3,0,1,2,3"
# Every line goes to classification/results/flif-v1-sleep4-pool.log (local).
set -u
GPUS=${1:-0,1,2,3,0,1,2,3}; PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -f scripts/gpu_pool2.sh ] || { echo "run from NSMT/"; exit 1; }
C=$PWD/nsmt_flif_v1/classification; PL=$C/results/flif-v1-sleep4-pool.log; R=nsmt_flif_v1/classification/scripts/run_sleep.sh
Q=$C/scripts/queues/sleep4-$(date +%Y%m%d).txt; S=flif-v1-sleep4
say() { echo "[sleep4 $(date '+%F %T')] $*" | tee -a "$PL"; }
[ -e "$C/results/$S/raw.txt" ] && { say "STOP: results/$S/raw.txt exists"; exit 1; }
mkdir -p "$(dirname "$Q")"; : > "$Q"
for f in 0 1 2 3 4; do
  for c in q_lif kv_lif; do for lr in 0.001 0.003; do echo "SUITE=$S LR=$lr bash $R $c $f" >> "$Q"; done; done
  for c in q_lif_mem_acc q_lif_mem q_lif_acc q_lif_mem_cshuf q_lif_mem_nml q_lif_acc_cshuf q_lif_cshuf q_lif_nml q_flif q_flif_mem; do
    echo "SUITE=$S LR=0.003 bash $R $c $f" >> "$Q"; done
done
N=$(grep -c . "$Q"); say "start: $N jobs, sha $(git rev-parse --short HEAD)"
bash scripts/gpu_pool2.sh "$Q" "$GPUS" 2>&1 | tee -a "$PL"
r=$(grep -c '^RESULT' "$C/results/$S/raw.txt" 2>/dev/null); k=$(grep -c '^SKIP' "$C/results/$S/raw.txt" 2>/dev/null)
say "RESULT ${r:-0} / $N, SKIP ${k:-0}"
if [ "${r:-0}" -ne "$N" ] || [ "${k:-0}" -ne 0 ]; then say "STOP: incomplete, no summary"; exit 1; fi
(cd "$C" && "$PY" summarize_sleep4.py) 2>&1 | tee -a "$PL"
say "ALL DONE"
