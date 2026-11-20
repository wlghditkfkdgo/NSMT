#!/usr/bin/env bash
# ettune_x_driver.sh (2026-11-20, exp/f-lif-pop-v3; PROJECT_LOG ETT tuning extension pre-registration; user decisions 11-20: patch up to 32,
# weight decay within the related-work range 0-0.06, 0.1 dropped). The stepwise tuning of the LIF SpkMixer again with the wider lists
# (ET_CHAIN=x): patch {4, 8*, 16, 24, 32} -> d_model -> MLP ratio -> blocks -> wd {0, 0.005, 0.01, 0.02, 0.05, 0.06*}, validation only,
# the same rules as ettune_driver.sh. Runs that already exist (11-01 default rows, flif-v1-ett-spkmux-ettune, the new suite) are not
# queued again, so a stage may have 0 new jobs. For each stage:
#   make_queue_ettune.sh <stage> (ET_CHAIN=x) -> gpu_pool2 (only when the queue is not empty) -> RESULT grew by the queue length, SKIP 0
#   -> summarize_ettune.py --stage <stage> -> recheck_ettune.py --stage <stage>
# then energy_recount_ett.py on the new suite -> summarize_ettune.py --stage final -> recheck final -> "driver end".
# Fail closed: any failure stops the driver (driver.log "STOP: ..."). Logging and commits stay with the session.
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/ettune_x_driver.sh > /dev/null 2>&1 &
#   FROM=<stage> resumes at that stage (its queue holds only the runs that are still missing)
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # NSMT
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
export ET_CHAIN=x
SU=flif-v1-ett-spkmux-ettune2
R="$C/results/$SU"; mkdir -p "$R"; LOGF="$R/driver.log"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
pyc() { (cd "$C" && LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B -u "$@" >> "$LOGF" 2>&1); }
cnt() { local n; n=$(grep -c "^$1" "$R/raw.txt" 2>/dev/null); echo "${n:-0}"; }   # 0 when raw.txt does not exist yet
STAGES=(patch dmodel mlp blocks wd)
say "driver start (ET_CHAIN=x, slots $SLOTS, from ${FROM:-patch})"
go=0
for st in "${STAGES[@]}"; do
  [ "$st" = "${FROM:-patch}" ] && go=1
  [ $go = 1 ] || { [ -s "$R/selected_$st.json" ] || die "FROM=${FROM:-} but selected_$st.json is missing"; continue; }
  qf=$(bash "$C/scripts/make_queue_ettune.sh" "$st" 2>>"$LOGF" | tail -1)
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
  pyc summarize_ettune.py --stage "$st" || die "selection $st failed"
  [ -s "$R/selected_$st.json" ] || die "no selected_$st.json"
  say "stage $st selected: $(python3 -c "import json; s=json.load(open('$R/selected_$st.json')); print(s['knob'], s['selected'], 'from', s['current'])")"
  pyc recheck_ettune.py --stage "$st" || die "independent recheck of $st failed"
done
[ "$(cnt RESULT)" -gt 0 ] || die "the new suite has no runs (the energy recount needs at least the patch stage's runs)"
(cd "$C" && CUDA_VISIBLE_DEVICES=${SLOTS%%,*} LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B -u energy_recount_ett.py --suite $SU >> "$LOGF" 2>&1) \
  || die "energy recount failed"
pyc summarize_ettune.py --stage final || die "final summary failed"
pyc recheck_ettune.py --stage final || die "independent recheck of final failed"
say "final: $(python3 -c "import json; f=json.load(open('$R/final.json')); print(f['config'], 'pass', f['pass']['tuned LIF SpkMixer'])")"
say "driver end"
