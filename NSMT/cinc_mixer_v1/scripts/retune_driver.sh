#!/usr/bin/env bash
# retune_driver.sh (2026-11-25, exp/f-lif-pop-v3; PROJECT_LOG pre-registration of 2026-11-25; user decision 11-25 "(나)"): ETTh1 and
# ETTm1 choose the LIF SpkMixer settings on their own validation with the procedure ETTh2 / ETTm2 went through (rt_plan.py), then the
# 16-cell judgment. Stages: lr -> layers -> patch -> dmodel -> mlp -> blocks -> wd (chains a and x together) -> final (H192 / H336).
# For each stage:
#   make_queue_rt.py <stage> -> gpu_pool2 (only when the queue is not empty) -> RESULT grew by the queue length, SKIP 0
#   -> summarize_rt.py --stage <stage> -> recheck_rt.py --stage <stage>        (final: no selection)
# then energy_recount_ett.py on the suite -> summarize_rt.py --stage judge -> recheck_rt.py --stage judge -> "driver end".
# Fail closed: any failure stops the driver (driver.log "STOP: ..."). Logging and commits stay with the session.
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/retune_driver.sh > /dev/null 2>&1 &
#   FROM=<stage> resumes at that stage (its queue holds only the runs that are still missing)
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # NSMT
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SU=flif-v1-ett-spkmux-retune
R="$C/results/$SU"; mkdir -p "$R"; LOGF="$R/driver.log"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
pyc() { (cd "$C" && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY -B -u "$@" >> "$LOGF" 2>&1); }
cnt() { local n; n=$(grep -c "^$1" "$R/raw.txt" 2>/dev/null); echo "${n:-0}"; }
STAGES=(lr layers patch dmodel mlp blocks wd final)
say "driver start (slots $SLOTS, from ${FROM:-lr}, HEAD $(cd "$N" && git rev-parse --short HEAD))"
go=0
for st in "${STAGES[@]}"; do
  [ "$st" = "${FROM:-lr}" ] && go=1
  [ $go = 1 ] || continue
  qf=$(cd "$C" && $PY -B scripts/make_queue_rt.py "$st" 2>>"$LOGF" | tail -1)
  case $qf in *:*) ;; *) die "queue for $st: $qf" ;; esac
  nq=${qf##*:}; qf=${qf%:*}
  case $nq in ''|*[!0-9]*) die "queue for $st: bad job count '$nq'" ;; esac
  [ -e "$qf" ] || die "queue file for $st missing ($qf)"
  [ "$(grep -c . "$qf")" -eq "$nq" ] || die "queue file for $st has $(grep -c . "$qf") lines, builder said $nq"
  before=$(cnt RESULT)
  if [ "$nq" -gt 0 ]; then
    say "stage $st: launched $nq jobs: ${qf#$N/}"
    (cd "$N" && bash scripts/gpu_pool2.sh "${qf#$N/}" "$SLOTS") > "$R/pool_log_$st.txt" 2>&1
    say "stage $st: pool done"
  else
    say "stage $st: 0 new jobs (every run of this stage exists)"
  fi
  nr=$(cnt RESULT); ns=$(cnt SKIP)
  say "stage $st: RESULT $nr (+$((nr - before)), want +$nq), SKIP $ns (want 0)"
  [ $((nr - before)) -eq "$nq" ] && [ "$ns" -eq 0 ] || die "incomplete stage $st"
  [ "$st" = final ] && continue
  pyc summarize_rt.py --stage "$st" || die "selection $st failed"
  say "stage $st selected: $(cd "$R" && for f in selected_${st}.json selected_a_${st}.json selected_x_${st}.json; do [ -s "$f" ] && python3 -c "import json; s=json.load(open('$f')); print('$f', s['selected'], 'from', s['current'], end='; ')"; done)"
  pyc recheck_rt.py --stage "$st" || die "independent recheck of $st failed"
done
(cd "$C" && CUDA_VISIBLE_DEVICES=${SLOTS%%,*} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY -B -u energy_recount_ett.py --suite $SU >> "$LOGF" 2>&1) \
  || die "energy recount failed"
pyc summarize_rt.py --stage judge || die "judge failed"
pyc recheck_rt.py --stage judge || die "independent recheck of judge failed"
say "judge: $(python3 -c "import json; j=json.load(open('$R/judge.json')); print({m: (round(v['mse16'], 4), round(v['mae16'], 4), v['verdict']) for m, v in j['models'].items()})")"
say "driver end"
