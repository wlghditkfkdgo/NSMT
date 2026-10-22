#!/usr/bin/env bash
# Cause-separation, light part (MODI_PROJECT_LOG 2026-10-22; user: start the cause analysis now). CinC 2017, 33 runs of the
# light conditions (position, recurrence, slow branches, hippocampus T=4, mixer soma init) next to the running idea-batch
# queue B and the linear-fusion runs: full-batch smoke of the 11 light conditions (any SKIP -> stop), wait for
# scripts/queues/fix_light.go (created after the scheduling entry is committed), one pool on all GPUs, then --fix summary.
# The heavy part (12 NSMT T=N runs) stays in rhythm_fix_driver.sh, which rhythm_bix_driver2.sh calls after queue B.
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
GPUS=${1:-0,1,2,3}; PY=/home/yschoi/.conda/envs/snn_recall/bin/python; L=/home/yschoi/.conda/envs/snn_recall/lib
Q=$W/scripts/queues; cd "$R"
log() { echo "[rhythm-fix-light $(date '+%F %T')] $*"; }
S=flif-v1-rhythm-fix-cinc; SM=smoke-rhythm-fix-light-$(date +%Y%m%d)
bash $W/scripts/make_queue_rhythm.sh $S-light CINC2017 fixlight 0.001 > /dev/null
sed "s/SUITE=$S-light/SUITE=$S/" $Q/$S-light.txt > $Q/rhythm-fix-light-now.txt
awk '!seen[$1]++' $Q/rhythm-fix-light-now.txt | sed -E "s/SUITE=\S+/SUITE=$SM/; s/LR=0.001 /LR=0.001 EPOCHS=1 /; s/run_rhythm2.sh [0-9]+$/run_rhythm2.sh 42/" > $Q/rhythm-fix-light-smoke.txt
log "light $(wc -l < $Q/rhythm-fix-light-now.txt) jobs, smoke $(wc -l < $Q/rhythm-fix-light-smoke.txt) jobs (suite $SM)"
bash scripts/gpu_pool2.sh $Q/rhythm-fix-light-smoke.txt "$GPUS" > $W/log/flif-v1-rhythm-fix-light-smoke.pool.log 2>&1
SR=$W/results/$SM/raw.txt
log "smoke: $(grep -c '^RESULT' $SR 2>/dev/null) RESULT, $(grep -c '^SKIP' $SR 2>/dev/null) SKIP"
if [ ! -s $SR ] || grep -q '^SKIP' $SR; then log "smoke failed -> light runs not started (report to the user)"; exit 1; fi
until [ -f $Q/fix_light.go ]; do sleep 60; done
log "go -> light pool on $GPUS"
bash scripts/gpu_pool2.sh $Q/rhythm-fix-light-now.txt "$GPUS" > $W/log/flif-v1-rhythm-fix-light.pool.log 2>&1
log "$S: $(grep -c '^RESULT' $W/results/$S/raw.txt) RESULT, $(grep -c '^SKIP' $W/results/$S/raw.txt) SKIP"
(cd $W && LD_LIBRARY_PATH=$L $PY summarize_rhythm.py --fix --suite $S > /dev/null) && log "summary results/$S/summary.md (partial until the heavy part ends)"
log "ALL DONE"
