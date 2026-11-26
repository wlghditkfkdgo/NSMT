#!/usr/bin/env bash
# sxa2_driver.sh (2026-11-27, exp/f-lif-pop-v3; PROJECT_LOG fusion-redesign stage-2 pre-registration; user decisions 11-27 "OT after
# Pearson", "MUX after no MUX"): 36 runs in the stage-1 selected form (results/flif-v1-ett-sxa-s1/selected_s1.json): OT plan mask, Pearson
# + MUX, OT + MUX; LIF, ADD, one block; ETTh1 / ETTm1 x H96 / H720 x seeds 42 / 7 / 2026; suite flif-v1-ett-sxa-s2:
#   queue (make_queue_sxa2.py) -> gpu_pool2.sh -> RESULT 36 / SKIP 0 -> energy_recount_ett.py -> summarize_sxa2.py
#   -> results/check-sxa-20261127/recheck_sxa2.py -> "driver end".  Fail closed: any failure stops the driver ("STOP: ...").
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/sxa2_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SU=flif-v1-ett-sxa-s2
R="$C/results/$SU"; mkdir -p "$R"; LOGF="$R/driver.log"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
cnt() { local n; n=$(grep -c "^$1" "$R/raw.txt" 2>/dev/null); echo "${n:-0}"; }
[ -e "$R/raw.txt" ] && die "$R/raw.txt exists (fresh suite expected)"
[ -s "$C/results/flif-v1-ett-sxa-s1/selected_s1.json" ] || die "stage-1 selection missing"
say "driver start (slots $SLOTS, HEAD $(cd "$N" && git rev-parse --short HEAD), stage-1 selection $(tr -d '\n ' < "$C/results/flif-v1-ett-sxa-s1/selected_s1.json"))"
qf=$(cd "$N" && $PY -B cinc_mixer_v1/scripts/make_queue_sxa2.py --suite $SU 2>>"$LOGF" | tail -1)
case $qf in *:36) ;; *) die "queue: $qf" ;; esac
qf=${qf%:*}
[ "$(grep -c . "$qf")" -eq 36 ] || die "queue file has $(grep -c . "$qf") lines"
say "launched 36 jobs: ${qf#$N/}"
(cd "$N" && bash scripts/gpu_pool2.sh "${qf#$N/}" "$SLOTS") > "$R/pool_log.txt" 2>&1
nr=$(cnt RESULT); ns=$(cnt SKIP)
say "pool done: RESULT $nr (want 36), SKIP $ns (want 0)"
[ "$nr" -eq 36 ] && [ "$ns" -eq 0 ] || die "incomplete"
(cd "$C" && CUDA_VISIBLE_DEVICES=${SLOTS%%,*} OMP_NUM_THREADS=4 $PY -B -u energy_recount_ett.py --suite $SU >> "$LOGF" 2>&1) || die "energy recount failed"
(cd "$C" && OMP_NUM_THREADS=4 $PY -B -u summarize_sxa2.py --suite $SU >> "$LOGF" 2>&1) || die "summary failed"
(cd "$C" && OMP_NUM_THREADS=4 $PY -B -u results/check-sxa-20261127/recheck_sxa2.py --suite $SU >> "$LOGF" 2>&1) || die "independent recheck failed"
say "driver end"
