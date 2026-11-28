#!/usr/bin/env bash
# mitres_driver.sh (2026-11-28, exp/f-lif-pop-v3; PROJECT_LOG pre-registration; user "1과 2를 진행해줘"):
#   item 1 (MIT-BIH SpkMixer energy): residual IAND and MS (membrane shortcut), lr {0.0025, 0.005, 0.01, 0.02} x seeds 42 / 7 / 2026 on
#          validation fold 0 (24 runs) -> select_mitres.py (highest seed-mean last-epoch validation macro-F1, ties keep 0.02) -> the
#          selected lr of each retrained on the whole training file (6 runs); suite flif-v1-mitbih-res
#   item 2 (order-only control): the ADD base (muxlin, lr 0.02) with TSHUF=1 (random token order per sample in training and
#          evaluation), seeds 42 / 7 / 2026 (3 runs); suite flif-v1-mitbih-tshuf
#   stage A queue (27) -> gpu_pool2.sh -> RESULT 24 + 3 / SKIP 0 -> select -> stage B queue (6) -> RESULT 30 / SKIP 0 ->
#   energy_recount_mitbih.py x 2 -> tshuf_spread.py -> summarize_mitres.py, summarize_tshuf.py -> "driver end". Fail closed ("STOP: ...").
# usage (from NSMT/): SLOTS=0,1,2,3,0,1,2,3 nohup bash cinc_mixer_v1/scripts/mitres_driver.sh > /dev/null 2>&1 &
set -u
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
C="$N/cinc_mixer_v1"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SLOTS=${SLOTS:-0,1,2,3,0,1,2,3}
SR=flif-v1-mitbih-res; ST=flif-v1-mitbih-tshuf
RR="$C/results/$SR"; RT="$C/results/$ST"; mkdir -p "$RR" "$RT"; LOGF="$RR/driver.log"
OR="$C/results/check-mitres-20261128"; OT="$C/results/check-tshuf-20261128"
RUN="bash cinc_mixer_v1/scripts/run_mitbih_spk.sh"
say() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
die() { say "STOP: $*"; exit 1; }
cnt() { local n; n=$(grep -c "^$2" "$1/raw.txt" 2>/dev/null); echo "${n:-0}"; }
[ -e "$RR/raw.txt" ] && die "$RR/raw.txt exists (fresh suite expected)"
[ -e "$RT/raw.txt" ] && die "$RT/raw.txt exists (fresh suite expected)"
[ -e "$RR/selected_lr.json" ] && die "selected_lr.json exists"
for f in "$OR/summary_mitres.json" "$OT/summary_tshuf.json" "$OT/tshuf_spread.json"; do [ -e "$f" ] && die "$f exists"; done
D=$(date +%Y%m%d)
QA="$C/scripts/queues/$SR-A-$D.txt"; QB="$C/scripts/queues/$SR-B-$D.txt"
[ -s "$QA" ] || [ -s "$QB" ] && die "queue exists"
: > "$QA"
for s in 42 7 2026; do echo "SUITE=$ST COND=spkmx_lif_muxlin_add LR=0.02 TSHUF=1 $RUN $s" >> "$QA"; done
for res in iand ms; do for lr in 0.0025 0.005 0.01 0.02; do for s in 42 7 2026; do
  echo "SUITE=$SR COND=spkmx_lif_muxlin_$res LR=$lr VAL_FOLD=0 $RUN $s" >> "$QA"
done; done; done
[ "$(grep -c . "$QA")" -eq 27 ] && [ "$(sort -u "$QA" | grep -c .)" -eq 27 ] || die "queue A: $(grep -c . "$QA") lines"
say "driver start (slots $SLOTS, HEAD $(cd "$N" && git rev-parse --short HEAD)): stage A 27 jobs ${QA#$N/}"
(cd "$N" && bash scripts/gpu_pool2.sh "${QA#$N/}" "$SLOTS") > "$RR/pool_log_A.txt" 2>&1
na=$(cnt "$RR" RESULT); ka=$(cnt "$RR" SKIP); nt=$(cnt "$RT" RESULT); kt=$(cnt "$RT" SKIP)
say "stage A done: res RESULT $na (want 24), SKIP $ka; tshuf RESULT $nt (want 3), SKIP $kt (want 0)"
[ "$na" -eq 24 ] && [ "$ka" -eq 0 ] && [ "$nt" -eq 3 ] && [ "$kt" -eq 0 ] || die "stage A incomplete"
(cd "$C" && $PY -B -u results/check-mitres-20261128/select_mitres.py >> "$LOGF" 2>&1) || die "selection failed"
ENV=$(cd "$C" && $PY -B results/check-mitres-20261128/select_mitres.py --print_env) || die "selection (env) failed"
LI=$(echo "$ENV" | grep '^IAND=' | cut -d= -f2); LM=$(echo "$ENV" | grep '^MS=' | cut -d= -f2)
case "$LI" in 0.0025|0.005|0.01|0.02) ;; *) die "bad IAND lr '$LI'" ;; esac
case "$LM" in 0.0025|0.005|0.01|0.02) ;; *) die "bad MS lr '$LM'" ;; esac
say "selected: IAND lr $LI, MS lr $LM"
: > "$QB"
for s in 42 7 2026; do echo "SUITE=$SR COND=spkmx_lif_muxlin_iand LR=$LI $RUN $s" >> "$QB"; done
for s in 42 7 2026; do echo "SUITE=$SR COND=spkmx_lif_muxlin_ms LR=$LM $RUN $s" >> "$QB"; done
[ "$(sort -u "$QB" | grep -c .)" -eq 6 ] || die "queue B"
say "stage B start: 6 jobs ${QB#$N/}"
(cd "$N" && bash scripts/gpu_pool2.sh "${QB#$N/}" "$SLOTS") > "$RR/pool_log_B.txt" 2>&1
nb=$(cnt "$RR" RESULT); kb=$(cnt "$RR" SKIP)
say "stage B done: res RESULT $nb (want 30), SKIP $kb (want 0)"
[ "$nb" -eq 30 ] && [ "$kb" -eq 0 ] || die "stage B incomplete"
G=${SLOTS%%,*}
(cd "$C" && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 $PY -B -W ignore -u energy_recount_mitbih.py --suite $SR >> "$LOGF" 2>&1) || die "res energy recount failed"
(cd "$C" && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 $PY -B -W ignore -u energy_recount_mitbih.py --suite $ST >> "$LOGF" 2>&1) || die "tshuf energy recount failed"
(cd "$C" && CUDA_VISIBLE_DEVICES=$G OMP_NUM_THREADS=4 $PY -B -W ignore -u results/check-tshuf-20261128/tshuf_spread.py --out "$OT/tshuf_spread.json" > "$OT/tshuf_spread.txt" 2>&1) || die "spread failed"
(cd "$C" && OMP_NUM_THREADS=4 $PY -B -u results/check-mitres-20261128/summarize_mitres.py >> "$LOGF" 2>&1) || die "summary (res) failed"
(cd "$C" && OMP_NUM_THREADS=4 $PY -B -u results/check-tshuf-20261128/summarize_tshuf.py >> "$LOGF" 2>&1) || die "summary (tshuf) failed"
say "driver end"
