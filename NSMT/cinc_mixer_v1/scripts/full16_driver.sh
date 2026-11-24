#!/usr/bin/env bash
# full16_driver.sh (2026-11-24, exp/f-lif-pop-v3; PROJECT_LOG full-ETT-table pre-registration; user decision 11-24 "두 개다 고려한채로
# 확장하자"): both final LIF SpkMixer configurations (first-chain final, extension final) on the 12 ETT cells not trained yet
# (ETTh1 / ETTm1 x H 96 / 192 / 336 / 720, ETTh2 / ETTm2 x H 192 / 336), seeds 42 / 7 / 2026 = 72 runs, suite flif-v1-ett-spkmux-full16.
# ETTh1 takes the ETTh2 settings and ETTm1 the ETTm2 settings (no selection on the new cells). Then
#   RESULT 72 / SKIP 0 -> energy_recount_ett.py -> SpikF syops counts for any dataset still missing (count_spikf16.py, no training)
#   -> results/ett-full16-20261124/summarize_full16.py -> "driver end".
# Fail closed: any failure stops (driver.log "STOP: ..."). Logging and commits stay with the session.
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/full16_driver.sh > /dev/null 2>&1 &
#        QUEUE_ONLY=1 ... : build and check the queue, then stop (no training)
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SU=flif-v1-ett-spkmux-full16
R="$C/results/$SU"; mkdir -p "$R"; LOGF="$R/driver.log"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
pyc() { (cd "$C" && LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B -u "$@" >> "$LOGF" 2>&1); }
[ -e "$R/raw.txt" ] && die "$R/raw.txt exists (fresh suite expected)"
Q="$C/scripts/queues/full16-20261124.txt"; : > "$Q"
for ds in ETTm1 ETTm2 ETTh1 ETTh2; do                     # slow datasets first
  case $ds in ETTh1|ETTm1) pls="96 192 336 720" ;; *) pls="192 336" ;; esac
  for cfg in ext first; do
    case ${ds:0:4}$cfg in
      ETThfirst) kn="LR=0.0005 LAYERS=1 PATCH=16 WD=0.1" ;;
      ETTmfirst) kn="LR=0.0001 LAYERS=1 PATCH=16 MLPR=2 WD=0.1" ;;
      ETThext)   kn="LR=0.0005 LAYERS=1 PATCH=24 WD=0.005" ;;
      ETTmext)   kn="LR=0.0001 LAYERS=1 PATCH=24 EMB=256 BLOCKS=2 WD=0.02" ;;
    esac
    for pl in $pls; do for s in 42 7 2026; do
      echo "SUITE=$SU COND=spkmx_lif_muxlin_iand $kn bash cinc_mixer_v1/scripts/run_ett_spk.sh $ds $pl $s" >> "$Q"
    done; done
  done
done
[ "$(grep -c . "$Q")" -eq 72 ] || die "queue has $(grep -c . "$Q") lines, want 72"
[ "$(sort -u "$Q" | grep -c .)" -eq 72 ] || die "queue lines are not distinct"
# distinct stdout / run-folder names: the runner names them by dataset, horizon, tag (set by the knobs) and seed
[ "$(sed -E 's/^SUITE=[^ ]+ //; s/ bash [^ ]+ / /' "$Q" | sort -u | grep -c .)" -eq 72 ] || die "queue keys (knobs, dataset, horizon, seed) are not distinct"
[ "${QUEUE_ONLY:-0}" = 1 ] && { say "queue only: 72 lines ${Q#$N/}"; exit 0; }
say "driver start (slots $SLOTS): 72 jobs ${Q#$N/}"
(cd "$N" && bash scripts/gpu_pool2.sh "${Q#$N/}" "$SLOTS") > "$R/pool_log.txt" 2>&1
nr=$(grep -c "^RESULT" "$R/raw.txt" 2>/dev/null); ns=$(grep -c "^SKIP" "$R/raw.txt" 2>/dev/null)
say "pool done: RESULT ${nr:-0} (want 72), SKIP ${ns:-0} (want 0)"
[ "${nr:-0}" -eq 72 ] && [ "${ns:-0}" -eq 0 ] || die "incomplete"
(cd "$C" && CUDA_VISIBLE_DEVICES=${SLOTS%%,*} LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B -u energy_recount_ett.py --suite $SU >> "$LOGF" 2>&1) \
  || die "energy recount failed"
for ds in ETTh1 ETTh2 ETTm1 ETTm2; do
  [ -s "$C/results/ett-full16-20261124/spikf16/spikf_syops_$ds.csv" ] && continue
  say "SpikF count $ds"
  (cd "$C" && CUDA_VISIBLE_DEVICES=${SLOTS%%,*} LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib" $PY -B -u results/ett-full16-20261124/count_spikf16.py --ds $ds >> "$LOGF" 2>&1) \
    || die "SpikF count $ds failed"
done
pyc results/ett-full16-20261124/summarize_full16.py || die "summary failed"
say "driver end"
