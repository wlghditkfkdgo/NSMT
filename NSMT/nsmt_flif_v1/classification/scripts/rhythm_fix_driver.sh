#!/usr/bin/env bash
# Cause-separation driver (MODI_PROJECT_LOG 2026-10-21): 1-B position, 1-A reach, 1-C hippocampus T=4, 3 population soma init.
# Both pools start only after log/flif-v1-rhythm-aux.driver.log says ALL DONE (the aux-loss population / f-LIF NSMT jobs take
# ~40 GB per GPU, so no second job fits beside them -- the smoke OOMed when tried).
#   heavy pool (NSMT T=N, ~35 GB): one per GPU; light pool A (mixers, SSA, hippocampus T=4): one per GPU beside it;
#   when the heavy pool drains, light pool B pops the same light queue (shared flock) so the light jobs run two per GPU.
#   then: summarize_rhythm.py --fix for both suites. Run inside tmux.
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
GPUS=${1:-0,1,2,3}; PY=/home/yschoi/.conda/envs/snn_recall/bin/python; L=/home/yschoi/.conda/envs/snn_recall/lib
Q=$W/scripts/queues; cd "$R"
log() { echo "[rhythm-fix $(date '+%F %T')] $*"; }
: > $Q/rhythm-fix-light.txt; : > $Q/rhythm-fix-heavy.txt
for D in SHD CINC2017; do
  S=flif-v1-rhythm-fix-$( [ $D = CINC2017 ] && echo cinc || echo shd )
  bash $W/scripts/make_queue_rhythm.sh $S-light $D fixlight 0.001 > /dev/null; sed "s/SUITE=$S-light/SUITE=$S/" $Q/$S-light.txt >> $Q/rhythm-fix-light.txt
  bash $W/scripts/make_queue_rhythm.sh $S-heavy $D fixheavy 0.001 > /dev/null; sed "s/SUITE=$S-heavy/SUITE=$S/" $Q/$S-heavy.txt >> $Q/rhythm-fix-heavy.txt
done
log "light $(wc -l < $Q/rhythm-fix-light.txt) jobs, heavy $(wc -l < $Q/rhythm-fix-heavy.txt) jobs; waiting for the aux-loss pool"
until grep -q 'ALL DONE' $W/log/flif-v1-rhythm-aux.driver.log 2>/dev/null; do sleep 300; done
log "aux-loss pool finished -> full-batch smoke (fixsmoke, 1 epoch, one per GPU), then heavy pool + light pool A"
: > $Q/rhythm-fix-smoke.txt
for D in SHD CINC2017; do
  bash $W/scripts/make_queue_rhythm.sh smoke-fix-$D $D fixsmoke 0.001 > /dev/null; sed "s/SUITE=smoke-fix-$D/SUITE=smoke-rhythm-fix-$(date +%Y%m%d)/" $Q/smoke-fix-$D.txt >> $Q/rhythm-fix-smoke.txt
done
bash scripts/gpu_pool2.sh $Q/rhythm-fix-smoke.txt "$GPUS" > $W/log/flif-v1-rhythm-fix-smoke.pool.log 2>&1
SM=$W/results/smoke-rhythm-fix-$(date +%Y%m%d)/raw.txt
log "smoke: $(grep -c '^RESULT' $SM 2>/dev/null) RESULT, $(grep -c '^SKIP' $SM 2>/dev/null) SKIP"
if grep -q '^SKIP' $SM 2>/dev/null; then log "smoke failed -> not starting the main pools (report to the user)"; exit 1; fi
bash scripts/gpu_pool2.sh $Q/rhythm-fix-light.txt "$GPUS" > $W/log/flif-v1-rhythm-fix-lightA.pool.log 2>&1 &
(
  bash scripts/gpu_pool2.sh $Q/rhythm-fix-heavy.txt "$GPUS" > $W/log/flif-v1-rhythm-fix-heavy.pool.log 2>&1
  log "heavy pool drained -> light pool B (same queue)"
  bash scripts/gpu_pool2.sh $Q/rhythm-fix-light.txt "$GPUS" > $W/log/flif-v1-rhythm-fix-lightB.pool.log 2>&1
) &
wait
for S in flif-v1-rhythm-fix-cinc flif-v1-rhythm-fix-shd; do
  [ -f $W/results/$S/raw.txt ] || continue
  log "$S: $(grep -c '^RESULT' $W/results/$S/raw.txt) RESULT, $(grep -c '^SKIP' $W/results/$S/raw.txt) SKIP"
  (cd $W && LD_LIBRARY_PATH=$L $PY summarize_rhythm.py --fix --suite $S > /dev/null) && log "summary results/$S/summary.md"
done
log "ALL DONE"
