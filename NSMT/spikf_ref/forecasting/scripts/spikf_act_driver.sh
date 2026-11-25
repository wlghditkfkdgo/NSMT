#!/usr/bin/env bash
# spikf_act_driver.sh (2026-11-26, exp/f-lif-pop-v3; PROJECT_LOG SpikF-LIF pre-registration; user request 11-25): the 96 SpikF runs
# (4 ETT datasets x 4 horizons x 3 seeds x head activation gelu / lif, released settings, --fast_eval) -> RESULT grew by 96, SKIP 0
# -> summarize_spikf_act.py -> recheck_spikf_act.py -> "driver end". Fail closed: any failure stops the driver ("STOP: ...").
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash spikf_ref/forecasting/scripts/spikf_act_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"     # NSMT
F="$N/spikf_ref/forecasting"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SU=spikf-act-ett
R="$F/results/$SU"; mkdir -p "$R"; LOGF="$R/driver.log"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
cnt() { local n; n=$(grep -c "^$1" "$R/raw.txt" 2>/dev/null); echo "${n:-0}"; }
say "driver start (slots $SLOTS, HEAD $(cd "$N" && git rev-parse --short HEAD))"
qf=$(cd "$N" && $PY -B spikf_ref/forecasting/scripts/make_queue_spikf.py --suite $SU 2>>"$LOGF" | tail -1)
case $qf in *:96) ;; *) die "queue: $qf" ;; esac
qf=${qf%:*}
[ "$(grep -c . "$qf")" -eq 96 ] || die "queue file has $(grep -c . "$qf") lines"
before=$(cnt RESULT)
say "launched 96 jobs: ${qf#$N/}"
(cd "$N" && bash scripts/gpu_pool2.sh "${qf#$N/}" "$SLOTS") > "$R/pool_log.txt" 2>&1
nr=$(cnt RESULT); ns=$(cnt SKIP)
say "pool done: RESULT $nr (+$((nr - before)), want +96), SKIP $ns (want 0)"
[ $((nr - before)) -eq 96 ] && [ "$ns" -eq 0 ] || die "incomplete"
(cd "$F" && OMP_NUM_THREADS=4 $PY -B -u summarize_spikf_act.py --suite $SU >> "$LOGF" 2>&1) || die "summary failed"
(cd "$F" && OMP_NUM_THREADS=4 $PY -B -u results/check-spikf-act-20261125/recheck_spikf_act.py --suite $SU >> "$LOGF" 2>&1) || die "independent recheck failed"
say "driver end"
