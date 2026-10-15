#!/usr/bin/env bash
# Rerun of the two stage-2 runs that failed on the file-name limit, then their memory probe and the summaries.
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
PY=/home/yschoi/.conda/envs/snn_recall/bin/python; L=/home/yschoi/.conda/envs/snn_recall/lib; S=flif-v1-s2p10
cd "$R"
echo "[fix $(date '+%F %T')] rerun start"
bash scripts/gpu_pool2.sh "$W/scripts/queues/s2p10-fix.txt" 0,1 > "$W/log/$S.fix.pool.log" 2>&1
grep '^RESULT' "$W/results/$S/raw.txt" | grep 'dataset=ETTm2_p720 seed=2026' | grep 'cond=neo_flif_both regime=L[24]g0.05' | grep -oE 'dir=\S+' | cut -d= -f2 > "$W/log/$S.fix.dirs"
echo "[fix $(date '+%F %T')] reruns with RESULT: $(wc -l < "$W/log/$S.fix.dirs")"
i=0; while read -r d; do
  (cd "$W" && CUDA_VISIBLE_DEVICES=$i LD_LIBRARY_PATH=$L OMP_NUM_THREADS=4 $PY neo_memory_probe.py --run "$d" --windows 512 --weights best --out results/$S/memory_probe_bestval -nd 0 | grep PROBE) &
  i=$((i+1)); done < "$W/log/$S.fix.dirs"; wait
(cd "$W" && LD_LIBRARY_PATH=$L $PY summarize_flif.py --suite $S --weights best --probe_dir results/$S/memory_probe_bestval --stage2 > /dev/null \
   && LD_LIBRARY_PATH=$L $PY summarize_flif.py --suite $S --weights last --stage2 --summary_name summary_lastweights.md > /dev/null)
echo "[fix $(date '+%F %T')] ALL DONE"
