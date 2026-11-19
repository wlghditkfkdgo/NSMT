#!/usr/bin/env bash
# ettune_driver.sh (2026-11-20, exp/f-lif-pop-v3; PROJECT_LOG ETT tuning pre-registration): the stepwise tuning of the LIF SpkMixer
# without waiting for a session. For each stage patch -> dmodel -> mlp -> blocks -> wd:
#   make_queue_ettune.sh <stage> (needs the earlier selections) -> gpu_pool2 -> RESULT count = cumulative want (24 60 96 120 156), SKIP 0
#   -> summarize_ettune.py --stage <stage> (validation-only selection, writes selected_<stage>.json) -> recheck_ettune.py --stage <stage>
# then energy_recount_ett.py (every new run; recomputed MSE / syops body checks) -> summarize_ettune.py --stage final -> recheck final.
# Fail closed: any failure stops the driver (driver.log "STOP: ..."); nothing is selected by hand. Logging and commits stay with the session.
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/ettune_driver.sh > /dev/null 2>&1 &
#   FROM=<stage> resumes at that stage (its queue is built and run again only if its RESULT rows are not all there)
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # NSMT
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SU=flif-v1-ett-spkmux-ettune
R="$C/results/$SU"; mkdir -p "$R"; LOGF="$R/driver.log"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
pyc() { (cd "$C" && LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B -u "$@" >> "$LOGF" 2>&1); }
cnt() { local n; n=$(grep -c "^$1" "$R/raw.txt" 2>/dev/null); echo "${n:-0}"; }   # 0 when raw.txt does not exist yet
STAGES=(patch dmodel mlp blocks wd); WANT=(24 60 96 120 156)
say "driver start (slots $SLOTS, from ${FROM:-patch})"
go=0
for i in 0 1 2 3 4; do
  st=${STAGES[$i]}; want=${WANT[$i]}
  [ "$st" = "${FROM:-patch}" ] && go=1
  [ $go = 1 ] || { [ -s "$R/selected_$st.json" ] || die "FROM=${FROM:-} but selected_$st.json is missing"; continue; }
  if [ "$(cnt RESULT)" -lt "$want" ]; then
    qf=$(bash "$C/scripts/make_queue_ettune.sh" "$st" 2>>"$LOGF" | tail -1)
    nq=${qf##*:}; qf=${qf%:*}
    [ -s "$qf" ] || die "queue for $st missing or empty ($qf)"
    say "stage $st: launched $nq jobs: ${qf#$N/}"
    (cd "$N" && bash scripts/gpu_pool2.sh "${qf#$N/}" "$SLOTS") > "$R/pool_log_$st.txt" 2>&1
    say "stage $st: pool done"
  fi
  nr=$(cnt RESULT); ns=$(cnt SKIP)
  say "stage $st: RESULT ${nr:-0}, SKIP ${ns:-0} (want $want, 0)"
  [ "${nr:-0}" -eq "$want" ] && [ "${ns:-0}" -eq 0 ] || die "incomplete stage $st"
  pyc summarize_ettune.py --stage "$st" || die "selection $st failed"
  [ -s "$R/selected_$st.json" ] || die "no selected_$st.json"
  say "stage $st selected: $(python3 -c "import json; s=json.load(open('$R/selected_$st.json')); print(s['knob'], s['selected'], 'from', s['current'])")"
  pyc recheck_ettune.py --stage "$st" || die "independent recheck of $st failed"
done
(cd "$C" && CUDA_VISIBLE_DEVICES=${SLOTS%%,*} LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B -u energy_recount_ett.py --suite $SU >> "$LOGF" 2>&1) \
  || die "energy recount failed"
pyc summarize_ettune.py --stage final || die "final summary failed"
pyc recheck_ettune.py --stage final || die "independent recheck of final failed"
say "final: $(python3 -c "import json; f=json.load(open('$R/final.json')); print(f['config'], 'pass', f['pass']['tuned LIF SpkMixer'])")"
say "driver end"
