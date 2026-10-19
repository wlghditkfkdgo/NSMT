#!/usr/bin/env bash
# Stage 9 driver (MODI_PROJECT_LOG 2026-10-19). Run inside tmux.
#   A. in parallel: learning-rate selection (nsmt_lif, seed 42, lr 1e-3 / 3e-3; heavy pool, one job per GPU) and the memory
#      pre-check (mix_lif_bidir / mix_none with 8, 32 or all tokens, lr 1e-3; light pool)
#   B. summarize_rhythm.py --decide -> chosen lr and gate per dataset (results/flif-v1-rhythm-crop/decision.json)
#   C. main runs only for datasets whose gate passed: heavy jobs (NSMT, ~27-35 GB) one per GPU, light jobs (mixers, SSA,
#      floor) in a second pool on the same GPUs
#   D. summaries
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
GPUS=${1:-0,1,2,3}; PY=/home/yschoi/.conda/envs/snn_recall/bin/python; L=/home/yschoi/.conda/envs/snn_recall/lib
Q=$W/scripts/queues; cd "$R"
log() { echo "[rhythm $(date '+%F %T')] $*"; }
log "A: lr selection + memory pre-check"
: > $Q/rhythm-lr.txt; : > $Q/rhythm-crop.txt
for D in CINC2017 SHD; do
  bash $W/scripts/make_queue_rhythm.sh rhythm-lr-$D $D lr > /dev/null; sed "s/SUITE=rhythm-lr-$D/SUITE=flif-v1-rhythm-lr/" $Q/rhythm-lr-$D.txt >> $Q/rhythm-lr.txt
  bash $W/scripts/make_queue_rhythm.sh rhythm-crop-$D $D crop 0.001 > /dev/null; sed "s/SUITE=rhythm-crop-$D/SUITE=flif-v1-rhythm-crop/" $Q/rhythm-crop-$D.txt >> $Q/rhythm-crop.txt
done
bash scripts/gpu_pool2.sh $Q/rhythm-lr.txt "$GPUS" > $W/log/flif-v1-rhythm-lr.pool.log 2>&1 &
bash scripts/gpu_pool2.sh $Q/rhythm-crop.txt "$GPUS" > $W/log/flif-v1-rhythm-crop.pool.log 2>&1 &
wait
log "A done: lr $(grep -c '^RESULT' $W/results/flif-v1-rhythm-lr/raw.txt) RESULT, crop $(grep -c '^RESULT' $W/results/flif-v1-rhythm-crop/raw.txt) RESULT"
(cd $W && LD_LIBRARY_PATH=$L $PY summarize_rhythm.py --decide) > $W/log/rhythm-decision.txt 2>&1
cat $W/log/rhythm-decision.txt
: > $Q/rhythm-main-heavy.txt; : > $Q/rhythm-main-light.txt
for D in CINC2017 SHD; do
  PASS=$($PY -c "import json;d=json.load(open('$W/results/flif-v1-rhythm-crop/decision.json'))['$D'];print(int(d['gate_pass']), d['lr'])")
  set -- $PASS
  if [ "$1" != 1 ]; then log "$D: memory gate NOT passed -> no main runs (report to the user)"; continue; fi
  S=flif-v1-rhythm-$( [ $D = CINC2017 ] && echo cinc || echo shd )
  bash $W/scripts/make_queue_rhythm.sh $S $D main $2 > /dev/null
  grep -E 'COND=nsmt_' $Q/$S.txt >> $Q/rhythm-main-heavy.txt; grep -vE 'COND=nsmt_' $Q/$S.txt >> $Q/rhythm-main-light.txt
  log "$D: gate passed, lr $2, suite $S"
done
log "C: main runs: $(wc -l < $Q/rhythm-main-heavy.txt) heavy, $(wc -l < $Q/rhythm-main-light.txt) light"
bash scripts/gpu_pool2.sh $Q/rhythm-main-heavy.txt "$GPUS" > $W/log/flif-v1-rhythm-main-heavy.pool.log 2>&1 &
bash scripts/gpu_pool2.sh $Q/rhythm-main-light.txt "$GPUS" > $W/log/flif-v1-rhythm-main-light.pool.log 2>&1 &
wait
for S in flif-v1-rhythm-cinc flif-v1-rhythm-shd; do
  [ -f $W/results/$S/raw.txt ] || continue
  log "$S: $(grep -c '^RESULT' $W/results/$S/raw.txt) RESULT, $(grep -c '^SKIP' $W/results/$S/raw.txt) SKIP"
  (cd $W && LD_LIBRARY_PATH=$L $PY summarize_rhythm.py --suite $S > /dev/null) && log "summary results/$S/summary.md"
done
log "ALL DONE"
