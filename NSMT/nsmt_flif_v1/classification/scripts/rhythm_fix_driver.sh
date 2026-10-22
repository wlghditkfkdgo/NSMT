#!/usr/bin/env bash
# Cause-separation driver (MODI_PROJECT_LOG 2026-10-21): 1-B position, 1-A reach, 1-C hippocampus T=4, 3 population soma init.
# CinC 2017 only since 10-21 23:3x (user: SHD is no longer a task -- its native time bins make T = N the standard there).
# 10-22: HEAVY PART ONLY (12 NSMT T=N runs; the light 33 runs run in rhythm_fix_light_driver.sh). Below, the original note:
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
for D in CINC2017; do                      # SHD dropped (user, 10-21: T = N claims need a non-spiking-native task)
  S=flif-v1-rhythm-fix-$( [ $D = CINC2017 ] && echo cinc || echo shd )
  bash $W/scripts/make_queue_rhythm.sh $S-light $D fixlight 0.001 > /dev/null; sed "s/SUITE=$S-light/SUITE=$S/" $Q/$S-light.txt >> $Q/rhythm-fix-light.txt
  bash $W/scripts/make_queue_rhythm.sh $S-heavy $D fixheavy 0.001 > /dev/null; sed "s/SUITE=$S-heavy/SUITE=$S/" $Q/$S-heavy.txt >> $Q/rhythm-fix-heavy.txt
done
log "heavy $(wc -l < $Q/rhythm-fix-heavy.txt) jobs (light part runs separately in rhythm_fix_light_driver.sh since 10-22)"
until grep -q 'ALL DONE' $W/log/flif-v1-rhythm-aux.driver.log 2>/dev/null; do sleep 300; done
SM=smoke-rhythm-fix-heavy-$(date +%Y%m%d)
awk '!seen[$1]++' $Q/rhythm-fix-heavy.txt | sed -E "s/SUITE=\S+/SUITE=$SM/; s/LR=0.001 /LR=0.001 EPOCHS=1 /; s/run_rhythm2.sh [0-9]+$/run_rhythm2.sh 42/" > $Q/rhythm-fix-smoke.txt
log "full-batch smoke of the heavy conditions: $(wc -l < $Q/rhythm-fix-smoke.txt) jobs (suite $SM)"
bash scripts/gpu_pool2.sh $Q/rhythm-fix-smoke.txt "$GPUS" > $W/log/flif-v1-rhythm-fix-smoke.pool.log 2>&1
SR=$W/results/$SM/raw.txt
log "smoke: $(grep -c '^RESULT' $SR 2>/dev/null) RESULT, $(grep -c '^SKIP' $SR 2>/dev/null) SKIP"
if [ ! -s $SR ] || grep -q '^SKIP' $SR; then log "smoke failed -> heavy runs not started (report to the user)"; exit 1; fi
bash scripts/gpu_pool2.sh $Q/rhythm-fix-heavy.txt "$GPUS" > $W/log/flif-v1-rhythm-fix-heavy.pool.log 2>&1
for S in flif-v1-rhythm-fix-cinc; do
  [ -f $W/results/$S/raw.txt ] || continue
  log "$S: $(grep -c '^RESULT' $W/results/$S/raw.txt) RESULT, $(grep -c '^SKIP' $W/results/$S/raw.txt) SKIP"
  (cd $W && LD_LIBRARY_PATH=$L $PY summarize_rhythm.py --fix --suite $S > /dev/null) && log "summary results/$S/summary.md"
done
log "ALL DONE"
