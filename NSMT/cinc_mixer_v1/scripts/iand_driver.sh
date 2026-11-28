#!/usr/bin/env bash
# iand_driver.sh (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "1. 1번 AD IAND 통일 진행해줘 2. 동시에 3번 순서 대조 다시하기"):
#   (1) AD residual unified to IAND: G5 settings (one mixer layer per direction, lr 2e-3) with IAND on MSL / SMAP x seeds 42 / 7 / 2026
#       (6 runs, suite flif-v1-ad-iand; PSM IAND = the 11-19 G5 runs) + 1 reproduction check run (PSM IAND seed 42, suite
#       check-adiand-repro, must equal the 11-19 row)
#   (2) order-only control on the MIT-BIH IAND base (muxlin IAND lr 0.01, TSHUF=1) x seeds 42 / 7 / 2026 (3 runs, suite
#       flif-v1-mitbih-tshuf-iand)
#   queue (10, slow first) -> gpu_pool2.sh -> RESULT 6 / 3 / 1, SKIP 0 -> energy_recount_mitbih.py -> tshuf_spread.py --tag ->
#   cls_order_eval.py --arm iand (IAND base, no training) -> summarize_adiand.py, summarize_tshufi.py -> recheck_iand.py -> "driver end".
#   Fail closed ("STOP: ...").
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/iand_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SA=flif-v1-ad-iand; ST=flif-v1-mitbih-tshuf-iand; SR=check-adiand-repro
RA="$C/results/$SA"; RT="$C/results/$ST"; RR="$C/results/$SR"; mkdir -p "$RA" "$RT" "$RR"; LOGF="$RA/driver.log"
O="$C/results/check-iand-20261128"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
cnt() { local n; n=$(grep -c "^$2" "$1/raw.txt" 2>/dev/null); echo "${n:-0}"; }
for d in "$RA" "$RT" "$RR"; do [ -e "$d/raw.txt" ] && die "$d/raw.txt exists (fresh suite expected)"; done
for f in summary_adiand.json summary_tshufi.json tshuf_spread_iand.json order_iand.json recheck_iand.txt; do [ -e "$O/$f" ] && die "$O/$f exists"; done
D=$(date +%Y%m%d); Q="$C/scripts/queues/iand-$D.txt"
[ -s "$Q" ] && die "queue exists"
AD="bash cinc_mixer_v1/scripts/run_ad_spk.sh"; MB="bash cinc_mixer_v1/scripts/run_mitbih_spk.sh"
: > "$Q"
for s in 42 7 2026; do echo "SUITE=$SA COND=spkmx_lif_muxlin_iand LR=0.002 MLAYERS=1 $AD SMAP $s" >> "$Q"; done
for s in 42 7 2026; do echo "SUITE=$ST COND=spkmx_lif_muxlin_iand LR=0.01 TSHUF=1 $MB $s" >> "$Q"; done
echo "SUITE=$SR COND=spkmx_lif_muxlin_iand LR=0.002 MLAYERS=1 $AD PSM 42" >> "$Q"
for s in 42 7 2026; do echo "SUITE=$SA COND=spkmx_lif_muxlin_iand LR=0.002 MLAYERS=1 $AD MSL $s" >> "$Q"; done
[ "$(grep -c . "$Q")" -eq 10 ] && [ "$(sort -u "$Q" | grep -c .)" -eq 10 ] || die "queue: $(grep -c . "$Q") lines"
say "driver start (slots $SLOTS, HEAD $(cd "$N" && git rev-parse --short HEAD)): 10 jobs ${Q#$N/}"
(cd "$N" && bash scripts/gpu_pool2.sh "${Q#$N/}" "$SLOTS") > "$RA/pool_log.txt" 2>&1
na=$(cnt "$RA" RESULT); ka=$(cnt "$RA" SKIP); nt=$(cnt "$RT" RESULT); kt=$(cnt "$RT" SKIP); nr=$(cnt "$RR" RESULT); kr=$(cnt "$RR" SKIP)
say "pool done: AD RESULT $na (want 6) SKIP $ka; MIT-BIH RESULT $nt (want 3) SKIP $kt; repro RESULT $nr (want 1) SKIP $kr"
[ "$na" -eq 6 ] && [ "$ka" -eq 0 ] && [ "$nt" -eq 3 ] && [ "$kt" -eq 0 ] && [ "$nr" -eq 1 ] && [ "$kr" -eq 0 ] || die "pool incomplete"
G=${SLOTS%%,*}
(cd "$C" && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 $PY -B -W ignore -u energy_recount_mitbih.py --suite $ST >> "$LOGF" 2>&1) || die "energy recount failed"
(cd "$C" && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 $PY -B -W ignore -u results/check-tshuf-20261128/tshuf_spread.py --suite $ST \
  --tag spkmx_lif_muxlin_iand_lr0.01_L2_tshuf --out "$O/tshuf_spread_iand.json" > "$O/tshuf_spread_iand.txt" 2>&1) || die "spread failed"
(cd "$C" && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 $PY -B -W ignore -u results/check-mitorder-20261128/cls_order_eval.py --arm iand \
  --out "$O/order_iand.json" > "$O/order_iand.txt" 2>&1) || die "order eval failed"
(cd "$C" && $PY -B -u results/check-iand-20261128/summarize_adiand.py >> "$LOGF" 2>&1) || die "summary (AD) failed"
(cd "$C" && $PY -B -u results/check-iand-20261128/summarize_tshufi.py >> "$LOGF" 2>&1) || die "summary (MIT-BIH) failed"
(cd "$C" && $PY -B -u results/check-iand-20261128/recheck_iand.py > "$O/recheck_iand.txt" 2>&1) || die "recheck failed"
grep -q 'ALL CHECKS PASSED' "$O/recheck_iand.txt" || die "recheck: not all checks passed"
say "driver end"
