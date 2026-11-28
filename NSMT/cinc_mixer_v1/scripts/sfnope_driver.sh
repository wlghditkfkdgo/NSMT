#!/usr/bin/env bash
# sfnope_driver.sh (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "SF-de의 위치 정보 끄기(ssa_pe 0): MIT-BIH 3 run,
# ETTh1/m1 12 run 수행하고 비교"): SF-de (COND=spk_ssa NALL=lif) with SSAPE=0 (--ssa_pe 0: no relative position bias in the SSA, tag
# _pe0); every other argument as the PE-on reference runs:
#   MIT-BIH lr 0.01, seeds 42 / 7 / 2026 (suite flif-v1-mitbih-sfde-nope, 3 runs; reference flif-v1-mitbih-sfde)
#   ETTh1 lr 5e-4 patch 8, ETTm1 lr 1e-4 patch 16, x H96 / H720 x seeds 42 / 7 / 2026 (suite flif-v1-ett-sfde-nope, 12 runs; reference
#   flif-v1-ett-sfde-h1m1)
#   queue (slow runs first) -> gpu_pool2.sh -> RESULT 3 + 12 / SKIP 0 -> energy_recount_mitbih.py / energy_recount_ett.py ->
#   cls_order_eval.py --arm sfdenope / order_eval.py --arm sfde1n (no training) -> results/check-sfnope-20261128/summarize_sfnope.py
#   -> "driver end".  Fail closed: any failure stops the driver ("STOP: ...").
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/sfnope_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SM=flif-v1-mitbih-sfde-nope; SE=flif-v1-ett-sfde-nope
RM="$C/results/$SM"; RE="$C/results/$SE"; mkdir -p "$RM" "$RE"; LOGF="$RE/driver.log"
O="$C/results/check-sfnope-20261128"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
cnt() { local n; n=$(grep -c "^$2" "$1/raw.txt" 2>/dev/null); echo "${n:-0}"; }
[ -e "$RM/raw.txt" ] && die "$RM/raw.txt exists (fresh suite expected)"
[ -e "$RE/raw.txt" ] && die "$RE/raw.txt exists (fresh suite expected)"
for f in order_sfdenope.json order_sfde1n.json summary_sfnope.json; do [ -e "$O/$f" ] && die "$O/$f exists"; done
Q="$C/scripts/queues/flif-v1-sfde-nope-$(date +%Y%m%d).txt"
[ -s "$Q" ] && die "queue $Q exists"
: > "$Q"
for s in 42 7 2026; do echo "SUITE=$SM COND=spk_ssa NALL=lif LR=0.01 SSAPE=0 bash cinc_mixer_v1/scripts/run_mitbih_spk.sh $s" >> "$Q"; done
for ds in ETTm1 ETTh1; do
  case $ds in ETTh1) kn="LR=0.0005" ;; ETTm1) kn="LR=0.0001 PATCH=16" ;; esac
  for pl in 720 96; do for s in 42 7 2026; do
    echo "SUITE=$SE COND=spk_ssa NALL=lif $kn SSAPE=0 bash cinc_mixer_v1/scripts/run_ett_spk.sh $ds $pl $s" >> "$Q"
  done; done
done
[ "$(grep -c . "$Q")" -eq 15 ] && [ "$(sort -u "$Q" | grep -c .)" -eq 15 ] || die "queue: $(grep -c . "$Q") lines"
say "driver start (slots $SLOTS, HEAD $(cd "$N" && git rev-parse --short HEAD)): 15 jobs ${Q#$N/}"
(cd "$N" && bash scripts/gpu_pool2.sh "${Q#$N/}" "$SLOTS") > "$RE/pool_log.txt" 2>&1
nm=$(cnt "$RM" RESULT); km=$(cnt "$RM" SKIP); ne=$(cnt "$RE" RESULT); ke=$(cnt "$RE" SKIP)
say "pool done: MIT-BIH RESULT $nm (want 3), SKIP $km; ETT RESULT $ne (want 12), SKIP $ke (want 0)"
[ "$nm" -eq 3 ] && [ "$km" -eq 0 ] && [ "$ne" -eq 12 ] && [ "$ke" -eq 0 ] || die "incomplete"
G=${SLOTS%%,*}
(cd "$C" && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 $PY -B -W ignore -u energy_recount_mitbih.py --suite $SM >> "$LOGF" 2>&1) || die "MIT-BIH energy recount failed"
(cd "$C" && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 $PY -B -W ignore -u energy_recount_ett.py --suite $SE >> "$LOGF" 2>&1) || die "ETT energy recount failed"
(cd "$C" && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 $PY -B -W ignore -u results/check-mitorder-20261128/cls_order_eval.py --arm sfdenope --out "$O/order_sfdenope.json" > "$O/order_sfdenope.txt" 2>&1) || die "MIT-BIH order eval failed"
(cd "$C" && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 $PY -B -W ignore -u results/check-order-20261127/order_eval.py --arm sfde1n --out "$O/order_sfde1n.json" > "$O/order_sfde1n.txt" 2>&1) || die "ETT order eval failed"
(cd "$C" && OMP_NUM_THREADS=4 $PY -B -u results/check-sfnope-20261128/summarize_sfnope.py >> "$LOGF" 2>&1) || die "summary failed"
say "driver end"
