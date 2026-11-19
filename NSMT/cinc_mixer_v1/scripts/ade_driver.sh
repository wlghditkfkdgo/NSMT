#!/usr/bin/env bash
# ade_driver.sh (2026-11-19, exp/f-lif-pop-v3; PROJECT_LOG AD energy pre-registration): the AD energy test without waiting for a session.
#   lrmean pool (9 (A) mean + 45 (B) lr holdout) -> completeness (54 RESULT, no SKIP) -> summarize_ade.py --stage lr (validation only)
#   -> res pool (18 holdout) -> completeness (72) -> summarize_ade.py --stage res -> final pool (9-18 whole file) -> completeness
#   -> summarize_ade.py --stage final
# Fail closed: any failure stops the driver. Logging and commits stay with the session.
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/ade_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # NSMT
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SU=flif-v1-ad-energy
R="$C/results/$SU"; mkdir -p "$R"; LOGF="$R/driver.log"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
pyc() { (cd "$C" && LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B -u "$@" >> "$LOGF" 2>&1); }
pool() {   # pool <stage> <jobs wanted>
  local qf; qf=$(bash "$C/scripts/make_queue_ade.sh" "$1" 2>>"$LOGF" | tail -1 | cut -d: -f1)
  [ -s "$qf" ] || die "queue missing or empty ($1: $qf)"
  [ "$(wc -l < "$qf")" -eq "$2" ] || die "queue $1 has $(wc -l < "$qf") lines, want $2"
  say "launched $2 jobs ($1): ${qf#$N/}"
  (cd "$N" && bash scripts/gpu_pool2.sh "${qf#$N/}" "$SLOTS") > "$R/pool_log_$1.txt" 2>&1
  say "pool done ($1)"
}
count() {  # count <RESULT rows wanted>
  local nr ns; nr=$(grep -c '^RESULT' "$R/raw.txt" 2>/dev/null); ns=$(grep -c '^SKIP' "$R/raw.txt" 2>/dev/null)
  say "$SU: RESULT ${nr:-0}, SKIP ${ns:-0} (want $1, 0)"
  [ "${nr:-0}" -eq "$1" ] && [ "${ns:-0}" -eq 0 ] || die "incomplete $SU"
}
say "driver start (slots $SLOTS)"
pool lrmean 54; count 54
pyc summarize_ade.py --stage lr || die "lr selection failed"
say "selected lr: $(tr -d '\n ' < "$R/selected_lr.json" | cut -c1-200)"
pool res 18; count 72
pyc summarize_ade.py --stage res || die "residual selection failed"
say "selected res: $(tr -d '\n ' < "$R/selected_res.json" | cut -c1-200)"
nf=$(python3 -c "import json; s = json.load(open('$R/selected_res.json'))['selected']; print(9 + 3 * sum(s['spike'][d] != s['res'][d] for d in s['res']))") || die "final job count"
pool final "$nf"; count $((72 + nf))
pyc summarize_ade.py --stage final || die "final summary failed"
say "driver end"
