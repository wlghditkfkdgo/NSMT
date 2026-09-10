#!/usr/bin/env bash
# Simple GPU pool: one job per GPU at a time, jobs popped atomically off a queue file.
# Each queue line is a command WITHOUT the trailing gpu argument -- the worker appends its own.
#   e.g.  bash neorecall_ad_v1/anomaly_detection/scripts/run_recall_ad.sh SMD recall 7      ->  bash neorecall_ad_v1/anomaly_detection/scripts/run_recall_ad.sh SMD recall 7 <gpu>
# usage: bash scripts/gpu_pool2.sh <queue_file> <gpu_csv>
# e.g. bash scripts/gpu_pool2.sh model_v1/forecasting/scripts/queues/trial.txt 0,1,2,3
# Keep queues under <model>/<task>/scripts/queues/; the .lock file lives beside the queue.
#
# WARNING: workers pop by in-place `sed -i '1d'`. Never rewrite the queue file from outside while
# a pool is running -- it races with the pop and silently drops jobs. To reorder, start a SECOND
# pool on a SEPARATE queue file and kill only the worker subshell of the GPU you want to free.
set -u
Q=${1:?queue file}; GPUS=${2:?gpu csv}
Q=$(readlink -f "$Q"); LOCK="$Q.lock"; : > "$LOCK"
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

pop() {                                   # echoes one command line, empty when queue is drained
  flock 9
  local n line
  n=$(grep -nvE '^[[:space:]]*(#|$)' "$Q" 2>/dev/null | head -1 | cut -d: -f1)
  [ -n "$n" ] || { echo ""; return; }
  line=$(sed -n "${n}p" "$Q")
  sed -i "${n}d" "$Q"
  echo "$line"
}

worker() {
  local g=$1
  while :; do
    local job
    job=$(pop 9<"$LOCK")
    [ -n "$job" ] || break
    echo "[pool gpu$g $(date +%H:%M:%S)] START  $job"
    ( cd "$R" && eval "$job $g" ) </dev/null
    echo "[pool gpu$g $(date +%H:%M:%S)] DONE(rc=$?)  $job"
  done
  echo "[pool gpu$g] queue drained, worker exit"
}

echo "[pool] queue=$Q gpus=$GPUS jobs=$(grep -cvE '^[[:space:]]*(#|$)' "$Q")"
pids=()
IFS=',' read -ra GARR <<< "$GPUS"
for g in "${GARR[@]}"; do worker "$g" & pids+=($!); echo "[pool] worker gpu$g pid=$!"; done
wait "${pids[@]}"
echo "[pool] all workers finished"
