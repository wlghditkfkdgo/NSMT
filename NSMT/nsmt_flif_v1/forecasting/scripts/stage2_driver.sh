#!/usr/bin/env bash
# Stage 2 (MODI_PROJECT_LOG 2026-10-15): train the 120-run queue, then the memory probe on every run, then the summary.
# Run from anywhere; meant to live inside a tmux session so it survives a dropped connection.
# usage: stage2_driver.sh [gpu_csv]   (default 0,1,2,3,0,1,2,3 = two workers per GPU)
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
GPUS=${1:-0,1,2,3,0,1,2,3}; PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$R"
echo "[driver $(date '+%F %T')] train start"
bash scripts/gpu_pool2.sh "$W/scripts/queues/flif-v1-s2.txt" "$GPUS" > "$W/log/flif-v1-s2.pool.log" 2>&1
echo "[driver $(date '+%F %T')] train done: $(grep -c '^RESULT' "$W/results/flif-v1-s2/raw.txt") RESULT, $(grep -c '^SKIP' "$W/results/flif-v1-s2/raw.txt") SKIP"
(cd "$W" && $PY make_post_queues.py --suite flif-v1-s2 --kinds probe)
bash scripts/gpu_pool2.sh "$W/scripts/queues/post-probe-flif-v1-s2.txt" "$GPUS" > "$W/log/flif-v1-s2.post-probe.pool.log" 2>&1
echo "[driver $(date '+%F %T')] probe done: $(ls "$W/results/flif-v1-s2/memory_probe/"*_probe.csv 2>/dev/null | wc -l) probe CSVs"
(cd "$W" && $PY summarize_flif.py --suite flif-v1-main --suite flif-v1-s2 --out_suite flif-v1-s2 \
   --probe_dir results/flif-v1-main/memory_probe --probe_dir results/flif-v1-s2/memory_probe --stage2 > /dev/null)
echo "[driver $(date '+%F %T')] summary written: results/flif-v1-s2/summary.md"
echo "[driver $(date '+%F %T')] ALL DONE"
