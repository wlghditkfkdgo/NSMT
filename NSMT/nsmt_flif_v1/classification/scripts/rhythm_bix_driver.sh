#!/usr/bin/env bash
# Idea-batch driver (MODI_PROJECT_LOG 2026-10-21): BiX / BiX-ch over the neuron axis, soma-less population mixers,
# population LIF mixer, Spikformer with attention BN (144 runs), then the cause-separation batch (rhythm_fix_driver.sh).
#   1. full-batch smoke now, one job per GPU next to the running aux-loss NSMT jobs (~40 GB each; mixer-level jobs need
#      < 8 GB); any SKIP stops the driver
#   2. wait for scripts/queues/bix.go (created after the pre-registration commit)
#   3. pool A: one worker per GPU on one queue (popk heavy runs first, then the rest)
#   4. when the aux-loss driver says ALL DONE: pool B (+1 worker per GPU); pool C (+1 more) if the smoke peak memory <= 14 GB
#   5. summaries (summarize_rhythm.py --bix), then bash rhythm_fix_driver.sh (cause-separation + population completions)
# Never edit this file or run_rhythm2.sh while it runs (bash reads scripts by byte offset).
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
GPUS=${1:-0,1,2,3}; PY=/home/yschoi/.conda/envs/snn_recall/bin/python; L=/home/yschoi/.conda/envs/snn_recall/lib
Q=$W/scripts/queues; cd "$R"
log() { echo "[rhythm-bix $(date '+%F %T')] $*"; }
DAY=$(date +%Y%m%d); SM=smoke-rhythm-bix-$DAY
: > $Q/rhythm-bix-smoke.txt; : > $Q/rhythm-bix.txt
for D in SHD CINC2017; do
  S=flif-v1-rhythm-bix-$( [ $D = CINC2017 ] && echo cinc || echo shd )
  bash $W/scripts/make_queue_rhythm.sh $SM-$D $D bixsmoke 0.001 > /dev/null; sed "s/SUITE=$SM-$D/SUITE=$SM/" $Q/$SM-$D.txt >> $Q/rhythm-bix-smoke.txt
  bash $W/scripts/make_queue_rhythm.sh $S-heavy $D bixheavy 0.001 > /dev/null; sed "s/SUITE=$S-heavy/SUITE=$S/" $Q/$S-heavy.txt >> $Q/rhythm-bix-heavy.tmp
done
for D in SHD CINC2017; do
  S=flif-v1-rhythm-bix-$( [ $D = CINC2017 ] && echo cinc || echo shd )
  bash $W/scripts/make_queue_rhythm.sh $S-light $D bixlight 0.001 > /dev/null; sed "s/SUITE=$S-light/SUITE=$S/" $Q/$S-light.txt >> $Q/rhythm-bix-light.tmp
done
cat $Q/rhythm-bix-heavy.tmp $Q/rhythm-bix-light.tmp > $Q/rhythm-bix.txt; rm -f $Q/rhythm-bix-heavy.tmp $Q/rhythm-bix-light.tmp
log "smoke $(wc -l < $Q/rhythm-bix-smoke.txt) jobs (suite $SM), main $(wc -l < $Q/rhythm-bix.txt) jobs (heavy first)"
bash scripts/gpu_pool2.sh $Q/rhythm-bix-smoke.txt "$GPUS" > $W/log/flif-v1-rhythm-bix-smoke.pool.log 2>&1
SR=$W/results/$SM/raw.txt
log "smoke: $(grep -c '^RESULT' $SR 2>/dev/null) RESULT, $(grep -c '^SKIP' $SR 2>/dev/null) SKIP"
if [ ! -s $SR ] || grep -q '^SKIP' $SR; then log "smoke failed -> stop (report to the user)"; exit 1; fi
PEAK=$(grep -h peak_mem_GB $W/log/$SM/*.stdout 2>/dev/null | awk '{if ($2 > m) m = $2} END {print m + 0}')
log "smoke peak memory ${PEAK} GB"
until [ -f $Q/bix.go ]; do sleep 60; done
log "go -> pool A (one per GPU)"
bash scripts/gpu_pool2.sh $Q/rhythm-bix.txt "$GPUS" > $W/log/flif-v1-rhythm-bix-A.pool.log 2>&1 &
(
  until grep -q 'ALL DONE' $W/log/flif-v1-rhythm-aux.driver.log 2>/dev/null; do sleep 300; done
  if [ "$(grep -cvE '^[[:space:]]*(#|$)' $Q/rhythm-bix.txt)" -gt 0 ]; then
    log "aux-loss pool finished -> pool B"
    bash scripts/gpu_pool2.sh $Q/rhythm-bix.txt "$GPUS" > $W/log/flif-v1-rhythm-bix-B.pool.log 2>&1 &
    if awk -v p="$PEAK" 'BEGIN {exit !(p <= 14)}'; then
      log "peak ${PEAK} GB <= 14 -> pool C"
      bash scripts/gpu_pool2.sh $Q/rhythm-bix.txt "$GPUS" > $W/log/flif-v1-rhythm-bix-C.pool.log 2>&1 &
    fi
    wait
  fi
) &
wait
for S in flif-v1-rhythm-bix-cinc flif-v1-rhythm-bix-shd; do
  [ -f $W/results/$S/raw.txt ] || continue
  log "$S: $(grep -c '^RESULT' $W/results/$S/raw.txt) RESULT, $(grep -c '^SKIP' $W/results/$S/raw.txt) SKIP"
  (cd $W && LD_LIBRARY_PATH=$L $PY summarize_rhythm.py --bix --suite $S > /dev/null) && log "summary results/$S/summary.md"
done
log "idea batch done -> cause-separation driver"
bash $W/scripts/rhythm_fix_driver.sh "$GPUS" > $W/log/flif-v1-rhythm-fix.driver.log 2>&1
log "ALL DONE"
