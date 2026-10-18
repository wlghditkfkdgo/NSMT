#!/usr/bin/env bash
# MIT-BIH suite flif-v1-cls (MODI_PROJECT_LOG 2026-10-19): 21 training runs, then cls_analysis.py on every run, then the
# summary with the pre-registered verdicts. Run inside tmux.
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
GPUS=${1:-0,1,2,3,0,1,2,3}; PY=/home/yschoi/.conda/envs/snn_recall/bin/python; L=/home/yschoi/.conda/envs/snn_recall/lib
S=${SUITE:-flif-v1-cls}
cd "$R"
echo "[cls $(date '+%F %T')] train start"
bash "$W/scripts/make_queue_cls.sh" $S
bash scripts/gpu_pool2.sh "$W/scripts/queues/$S.txt" "$GPUS" > "$W/log/$S.pool.log" 2>&1
echo "[cls $(date '+%F %T')] train done: $(grep -c '^RESULT' "$W/results/$S/raw.txt") RESULT, $(grep -c '^SKIP' "$W/results/$S/raw.txt") SKIP"
Q="$W/scripts/queues/analysis-$S.txt"; : > "$Q"
grep '^RESULT' "$W/results/$S/raw.txt" | while read -r line; do
  c=$(echo "$line" | grep -oE ' cond=\S+' | cut -d= -f2); s=$(echo "$line" | grep -oE ' seed=\S+' | cut -d= -f2)
  echo "bash nsmt_flif_v1/classification/scripts/run_analysis.sh $S $c $s" >> "$Q"; done
bash scripts/gpu_pool2.sh "$Q" "$GPUS" > "$W/log/$S.analysis.pool.log" 2>&1
echo "[cls $(date '+%F %T')] analysis done: $(ls -d "$W/results/$S/analysis/"*/ 2>/dev/null | wc -l) run folders, $(grep -c 'ANALYSIS-FAIL' "$W/log/$S.analysis.pool.log") failed"
(cd "$W" && LD_LIBRARY_PATH=$L $PY summarize_cls.py --suite $S > /dev/null) && echo "[cls $(date '+%F %T')] summary: results/$S/summary.md"
echo "[cls $(date '+%F %T')] ALL DONE"
