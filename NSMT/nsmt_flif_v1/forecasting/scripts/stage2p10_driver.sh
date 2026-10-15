#!/usr/bin/env bash
# Stage 2 with patience 10 (MODI_PROJECT_LOG 2026-10-15): train 168 runs, memory probe on the best-validation weights,
# summaries for best-validation (primary) and last-epoch (secondary) weights. Meant to run inside tmux.
# usage: stage2p10_driver.sh [gpu_csv]   (default 0,1,2,3,0,1,2,3 = two workers per GPU)
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
GPUS=${1:-0,1,2,3,0,1,2,3}; PY=/home/yschoi/.conda/envs/snn_recall/bin/python
L="$(dirname "$(dirname "$PY")")/lib"
S=flif-v1-s2p10
cd "$R"
echo "[driver $(date '+%F %T')] train start"
bash "$W/scripts/make_queue_s2p10.sh"
bash scripts/gpu_pool2.sh "$W/scripts/queues/$S.txt" "$GPUS" > "$W/log/$S.pool.log" 2>&1
echo "[driver $(date '+%F %T')] train done: $(grep -c '^RESULT' "$W/results/$S/raw.txt") RESULT, $(grep -c '^SKIP' "$W/results/$S/raw.txt") SKIP"
(cd "$W" && LD_LIBRARY_PATH=$L $PY make_post_queues.py --suite $S --kinds probe --weights best)
bash scripts/gpu_pool2.sh "$W/scripts/queues/post-probe-$S-bestval.txt" "$GPUS" > "$W/log/$S.post-probe.pool.log" 2>&1
echo "[driver $(date '+%F %T')] probe done: $(ls "$W/results/$S/memory_probe_bestval/"*_probe.csv 2>/dev/null | wc -l) probe CSVs"
(cd "$W" && LD_LIBRARY_PATH=$L $PY summarize_flif.py --suite $S --weights best --probe_dir results/$S/memory_probe_bestval --stage2 > /dev/null \
   && LD_LIBRARY_PATH=$L $PY summarize_flif.py --suite $S --weights last --stage2 --summary_name summary_lastweights.md > /dev/null)
echo "[driver $(date '+%F %T')] summaries: results/$S/summary.md (best-validation weights), summary_lastweights.md"
echo "[driver $(date '+%F %T')] ALL DONE"
