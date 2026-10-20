#!/usr/bin/env bash
# NSMT aux-loss options driver (MODI_PROJECT_LOG 2026-10-20). Run inside tmux.
#   options: auxoff (alpha 0, plain CE) / full (keep_ratio 0: reconstruct the full patched signal) / a05 / a09 (alpha 0.5 / 0.9)
#   x NSMT neurons lif, pop, flif x main seeds (CinC 42/7/2026 fold 0, SHD 42/7/2026/123/1024), lr 1e-3 (stage-9 choice)
#   one pool, one job per GPU (NSMT needs ~27-35 GB); queue order: neuron groups lif -> pop -> flif, each SHD then CinC
#   summaries: summarize_rhythm.py --aux (baseline alpha 0.1 + DCT and spk_ssa from the stage-9 suites)
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
GPUS=${1:-0,1,2,3}; PY=/home/yschoi/.conda/envs/snn_recall/bin/python; L=/home/yschoi/.conda/envs/snn_recall/lib
Q=$W/scripts/queues; cd "$R"
log() { echo "[rhythm-aux $(date '+%F %T')] $*"; }
bash $W/scripts/make_queue_rhythm.sh flif-v1-rhythm-aux-shd SHD aux 0.001 > /dev/null
bash $W/scripts/make_queue_rhythm.sh flif-v1-rhythm-aux-cinc CINC2017 aux 0.001 > /dev/null
: > $Q/rhythm-aux.txt
for n in nsmt_lif nsmt_pop nsmt_flif; do
  grep "COND=${n}_" $Q/flif-v1-rhythm-aux-shd.txt >> $Q/rhythm-aux.txt
  grep "COND=${n}_" $Q/flif-v1-rhythm-aux-cinc.txt >> $Q/rhythm-aux.txt
done
log "queue: $(wc -l < $Q/rhythm-aux.txt) jobs"
bash scripts/gpu_pool2.sh $Q/rhythm-aux.txt "$GPUS" > $W/log/flif-v1-rhythm-aux.pool.log 2>&1
for S in flif-v1-rhythm-aux-cinc flif-v1-rhythm-aux-shd; do
  [ -f $W/results/$S/raw.txt ] || continue
  log "$S: $(grep -c '^RESULT' $W/results/$S/raw.txt) RESULT, $(grep -c '^SKIP' $W/results/$S/raw.txt) SKIP"
  (cd $W && LD_LIBRARY_PATH=$L $PY summarize_rhythm.py --aux --suite $S > /dev/null) && log "summary results/$S/summary.md"
done
log "ALL DONE"
