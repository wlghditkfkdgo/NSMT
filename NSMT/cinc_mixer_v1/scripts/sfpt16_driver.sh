#!/usr/bin/env bash
# sfpt16_driver.sh (2026-11-23, exp/f-lif-pop-v3; PROJECT_LOG SF-de patch-16 pre-registration; user request 11-23 "(가) 실행"): SF-de (the
# Spikformer-style model of the criteria: COND=spk_ssa NALL=lif, selected lr ETTh2 5e-4 / ETTm2 1e-4, layers ETTh2 1 / ETTm2 2) with
# PATCH=16, ETTh2 / ETTm2 x H96 / H720 x seeds 42 / 7 / 2026 (12 runs, suite flif-v1-ett-sfde-pt16), then
#   RESULT 12 / SKIP 0 -> energy_recount_ett.py -> results/sfde-pt16-20261123/summarize_sfpt16.py -> energy-common table -> "driver end".
# Fail closed: any failure stops (driver.log "STOP: ..."). Logging and commits stay with the session.
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/sfpt16_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SU=flif-v1-ett-sfde-pt16
R="$C/results/$SU"; mkdir -p "$R"; LOGF="$R/driver.log"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
pyc() { (cd "$C" && LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B -u "$@" >> "$LOGF" 2>&1); }
[ -e "$R/raw.txt" ] && die "$R/raw.txt exists (fresh suite expected)"
Q="$C/scripts/queues/sfpt16-20261123.txt"; : > "$Q"
for ds in ETTm2 ETTh2; do
  case $ds in ETTh2) lr=0.0005; ly=1 ;; ETTm2) lr=0.0001; ly=2 ;; esac
  for pl in 96 720; do for s in 42 7 2026; do
    echo "SUITE=$SU COND=spk_ssa NALL=lif LR=$lr LAYERS=$ly PATCH=16 bash cinc_mixer_v1/scripts/run_ett_spk.sh $ds $pl $s" >> "$Q"
  done; done
done
[ "$(grep -c . "$Q")" -eq 12 ] || die "queue has $(grep -c . "$Q") lines, want 12"
say "driver start (slots $SLOTS): 12 jobs ${Q#$N/}"
(cd "$N" && bash scripts/gpu_pool2.sh "${Q#$N/}" "$SLOTS") > "$R/pool_log.txt" 2>&1
nr=$(grep -c "^RESULT" "$R/raw.txt" 2>/dev/null); ns=$(grep -c "^SKIP" "$R/raw.txt" 2>/dev/null)
say "pool done: RESULT ${nr:-0} (want 12), SKIP ${ns:-0} (want 0)"
[ "${nr:-0}" -eq 12 ] && [ "${ns:-0}" -eq 0 ] || die "incomplete"
(cd "$C" && CUDA_VISIBLE_DEVICES=${SLOTS%%,*} LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B -u energy_recount_ett.py --suite $SU >> "$LOGF" 2>&1) \
  || die "energy recount failed"
pyc results/sfde-pt16-20261123/summarize_sfpt16.py || die "summary failed"
(cd "$C" && LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B results/energy-common-20261123/common_table.py \
  > results/energy-common-20261123/summary_with_pt16.md 2>> "$LOGF") || die "energy table failed"
say "driver end"
