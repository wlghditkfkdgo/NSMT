#!/usr/bin/env bash
# Stage 3 Phase B main run (MODI_PROJECT_LOG 2026-10-17): 60 serial / plain-embedding runs, the memory probe on the serial runs
# (best-validation weights), and the summary with the stage-3 questions against the stage-2 references. Run inside tmux.
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
GPUS=${1:-0,1,2,3,0,1,2,3}; PY=/home/yschoi/.conda/envs/snn_recall/bin/python; L=/home/yschoi/.conda/envs/snn_recall/lib
S=flif-v1-s3
cd "$R"
echo "[s3 $(date '+%F %T')] train start"
bash "$W/scripts/make_queue_s3.sh"
bash scripts/gpu_pool2.sh "$W/scripts/queues/$S.txt" "$GPUS" > "$W/log/$S.pool.log" 2>&1
echo "[s3 $(date '+%F %T')] train done: $(grep -c '^RESULT' "$W/results/$S/raw.txt") RESULT, $(grep -c '^SKIP' "$W/results/$S/raw.txt") SKIP"
(cd "$W" && LD_LIBRARY_PATH=$L $PY make_post_queues.py --suite $S --kinds probe --weights best)
bash scripts/gpu_pool2.sh "$W/scripts/queues/post-probe-$S-bestval.txt" "$GPUS" > "$W/log/$S.post-probe.pool.log" 2>&1
echo "[s3 $(date '+%F %T')] probe done: $(ls "$W/results/$S/memory_probe_bestval/"*_probe.csv 2>/dev/null | wc -l) probe CSVs"
(cd "$W" && LD_LIBRARY_PATH=$L $PY summarize_flif.py --suite $S --suite flif-v1-s2p10 --out_suite $S --weights best --stage3 \
   --probe_dir results/$S/memory_probe_bestval > /dev/null \
 && LD_LIBRARY_PATH=$L $PY summarize_flif.py --suite $S --suite flif-v1-s2p10 --out_suite $S --weights last --stage3 \
   --summary_name summary_lastweights.md > /dev/null)
echo "[s3 $(date '+%F %T')] summaries: results/$S/summary.md (best-validation weights), summary_lastweights.md"
echo "[s3 $(date '+%F %T')] ALL DONE"
