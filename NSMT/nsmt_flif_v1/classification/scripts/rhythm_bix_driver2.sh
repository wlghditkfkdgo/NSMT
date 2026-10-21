#!/usr/bin/env bash
# Idea-batch driver, part 2 (MODI_PROJECT_LOG 2026-10-22; follows rhythm_bix_driver.sh, which ran the smoke and stopped on
# two out-of-memory SKIPs). CinC 2017 only (SHD dropped 10-21).
#   queue A (scripts/queues/rhythm-bix-A.txt): 30 runs whose calibration is settled -- after scripts/queues/bix.go:
#       pool A1 on the GPUs with memory left next to the aux-loss NSMT jobs ($GPUS_NOW, one each);
#       when the aux-loss driver says ALL DONE: pool A2 on all GPUs twice ("0,1,2,3,0,1,2,3")
#   queue B (scripts/queues/rhythm-bix-B.txt): popg / popk 24 runs -- after scripts/queues/bix_b.go (recalibration
#       pre-registered): full-batch smoke of the 8 conditions (any SKIP -> stop), then pool B on all GPUs twice
#   then summarize_rhythm.py --bix, then rhythm_fix_driver.sh (cause-separation, CinC 45 runs).
# Never edit this file, run_rhythm2.sh or a queue file in use while it runs.
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
GPUS_NOW=${1:-2,3}; ALL2=${2:-0,1,2,3,0,1,2,3}; PY=/home/yschoi/.conda/envs/snn_recall/bin/python; L=/home/yschoi/.conda/envs/snn_recall/lib
Q=$W/scripts/queues; cd "$R"
log() { echo "[rhythm-bix2 $(date '+%F %T')] $*"; }
aux_done() { grep -q 'ALL DONE' $W/log/flif-v1-rhythm-aux.driver.log 2>/dev/null; }
log "queue A $(grep -cvE '^\s*(#|$)' $Q/rhythm-bix-A.txt) jobs, queue B $(grep -cvE '^\s*(#|$)' $Q/rhythm-bix-B.txt) jobs; waiting for bix.go"
until [ -f $Q/bix.go ]; do sleep 60; done
log "go A -> pool A1 on $GPUS_NOW"
bash scripts/gpu_pool2.sh $Q/rhythm-bix-A.txt "$GPUS_NOW" > $W/log/flif-v1-rhythm-bix-A1.pool.log 2>&1 &
(
  until aux_done; do sleep 300; done
  if [ "$(grep -cvE '^\s*(#|$)' $Q/rhythm-bix-A.txt)" -gt 0 ]; then
    log "aux-loss pool finished -> pool A2 on $ALL2"
    bash scripts/gpu_pool2.sh $Q/rhythm-bix-A.txt "$ALL2" > $W/log/flif-v1-rhythm-bix-A2.pool.log 2>&1
  fi
) &
(
  until [ -f $Q/bix_b.go ]; do sleep 120; done
  until aux_done; do sleep 300; done                                   # popk needs the memory the NSMT jobs hold
  DAY=$(date +%Y%m%d); SM=smoke-rhythm-bix-b-$DAY
  awk '!seen[$1]++' $Q/rhythm-bix-B.txt | sed -E "s/SUITE=\S+/SUITE=$SM/; s/LR=0.001 /LR=0.001 EPOCHS=1 /; s/run_rhythm2.sh [0-9]+$/run_rhythm2.sh 42/" > $Q/rhythm-bix-B-smoke.txt
  log "go B -> smoke $(wc -l < $Q/rhythm-bix-B-smoke.txt) jobs (suite $SM)"
  bash scripts/gpu_pool2.sh $Q/rhythm-bix-B-smoke.txt "0,1,2,3" > $W/log/flif-v1-rhythm-bix-B-smoke.pool.log 2>&1
  SR=$W/results/$SM/raw.txt
  log "smoke B: $(grep -c '^RESULT' $SR 2>/dev/null) RESULT, $(grep -c '^SKIP' $SR 2>/dev/null) SKIP"
  if [ ! -s $SR ] || grep -q '^SKIP' $SR; then log "smoke B failed -> queue B not started (report to the user)"; exit 1; fi
  log "pool B on $ALL2"
  bash scripts/gpu_pool2.sh $Q/rhythm-bix-B.txt "$ALL2" > $W/log/flif-v1-rhythm-bix-B.pool.log 2>&1
) &
wait
for S in flif-v1-rhythm-bix-cinc; do
  [ -f $W/results/$S/raw.txt ] || continue
  log "$S: $(grep -c '^RESULT' $W/results/$S/raw.txt) RESULT, $(grep -c '^SKIP' $W/results/$S/raw.txt) SKIP"
  (cd $W && LD_LIBRARY_PATH=$L $PY summarize_rhythm.py --bix --suite $S > /dev/null) && log "summary results/$S/summary.md"
done
if [ "$(grep -cvE '^\s*(#|$)' $Q/rhythm-bix-B.txt)" -gt 0 ]; then log "queue B not finished -> cause-separation batch NOT started (report to the user)"; exit 1; fi
log "idea batch done -> cause-separation driver"
bash $W/scripts/rhythm_fix_driver.sh "0,1,2,3" > $W/log/flif-v1-rhythm-fix.driver.log 2>&1
log "ALL DONE"
