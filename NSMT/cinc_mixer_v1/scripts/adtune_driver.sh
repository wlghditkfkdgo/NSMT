#!/usr/bin/env bash
# adtune_driver.sh (2026-11-29, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "(가) epoch 수, 정규화, layer/block 개수, hidden_dim
# 개수 등등 하이퍼파라미터 튜닝", decisions "test 전체로 선택", "AUC-PR로 선택"): AD tuning of the LIF SpkMixer, stage by stage
# (adtune.py: norm -> epochs -> width -> depth -> patch -> lr -> res). Per stage: adtune.py --queue (missing runs only) -> gpu_pool2.sh ->
# RESULT +n / SKIP 0 -> adtune.py --select. Then summarize_adtune2.py -> recheck_adtune2.py -> "driver end". Fail closed ("STOP: ...").
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/adtune_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SU=flif-v1-ad-tune2; R="$C/results/$SU"; mkdir -p "$R"; LOGF="$R/driver.log"
O="$C/results/check-adtune2-20261129"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
cnt() { local n; n=$(grep -c "^$1" "$R/raw.txt" 2>/dev/null); echo "${n:-0}"; }
[ -e "$R/raw.txt" ] && die "$R/raw.txt exists (fresh suite expected)"
ls "$R"/selected_*.json > /dev/null 2>&1 && die "selection files exist"
for f in summary_adtune2.json recheck_adtune2.txt; do [ -e "$O/$f" ] && die "$O/$f exists"; done
D=$(date +%Y%m%d)
say "driver start (slots $SLOTS, HEAD $(cd "$N" && git rev-parse --short HEAD))"
for st in norm epochs width depth patch lr res; do
  Q="$C/scripts/queues/$SU-$st-$D.txt"
  [ -s "$Q" ] && die "queue $Q exists"
  (cd "$C" && $PY -B adtune.py --stage $st --queue "$Q" >> "$LOGF" 2>&1) || die "queue $st failed"
  n=$(grep -c . "$Q"); r0=$(cnt RESULT)
  [ "$(sort -u "$Q" | grep -c .)" -eq "$n" ] || die "queue $st: duplicate lines"
  say "stage $st: $n runs ${Q#$N/}"
  if [ "$n" -gt 0 ]; then
    (cd "$N" && bash scripts/gpu_pool2.sh "${Q#$N/}" "$SLOTS") > "$R/pool_log_$st.txt" 2>&1
  fi
  r1=$(cnt RESULT); k=$(cnt SKIP)
  say "stage $st done: RESULT +$((r1 - r0)) (want $n), SKIP $k (want 0)"
  [ $((r1 - r0)) -eq "$n" ] && [ "$k" -eq 0 ] || die "stage $st incomplete"
  (cd "$C" && $PY -B adtune.py --stage $st --select >> "$LOGF" 2>&1) || die "selection $st failed"
  say "stage $st selected: $(tr -d '\n ' < "$R/selected_$st.json" | grep -oE '"selected":\{[^}]*\}')"
done
(cd "$C" && $PY -B -u results/check-adtune2-20261129/summarize_adtune2.py >> "$LOGF" 2>&1) || die "summary failed"
(cd "$C" && $PY -B -u results/check-adtune2-20261129/recheck_adtune2.py > "$O/recheck_adtune2.txt" 2>&1) || die "recheck failed"
grep -q 'ALL CHECKS PASSED' "$O/recheck_adtune2.txt" || die "recheck: not all checks passed"
say "driver end"
