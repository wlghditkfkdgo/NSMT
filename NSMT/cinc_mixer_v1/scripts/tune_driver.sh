#!/usr/bin/env bash
# tune_driver.sh (2026-11-02, exp/f-lif-pop-v3): runs the remaining pre-registered stages of the fusion-MUX tuning (PROJECT_LOG
# 2026-11-01 16:56 / 18:4x) without waiting for a session: wait for the running pool -> validation-only selection
# (summarize_*_tune.py, writes selected_<stage>.json) -> next queue (make_queue_spk.sh) -> pool -> ... -> final summary.
# Fail closed: a stage whose selection file is not written (missing run, SKIP) stops the driver; nothing is selected by hand.
# Logging to docs/PROJECT_LOG.md and commits stay with the session (driver.log in the suite folder records every step).
# usage (from NSMT/): nohup bash cinc_mixer_v1/scripts/tune_driver.sh ett|mit|cinc > /dev/null 2>&1 &
set -u
TASK=${1:?ett|mit|cinc}
N="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"     # NSMT
C="$N/cinc_mixer_v1"; Q="$C/scripts/queues"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
case $TASK in
  ett)  SUITE=flif-v1-ett-spkmux-tune;     SUM=summarize_ett_tune.py;    SLOTS=0,1,2,3,0,1,2,3; STAGES="layers res" ;;
  mit)  SUITE=flif-v1-mitbih-spkmux-tune;  SUM=summarize_mitbih_tune.py; SLOTS=0,1,2,3;         STAGES="layers res" ;;
  cinc) SUITE=flif-v1-rhythm-spk-tune-cinc; SUM=summarize_cinc_tune.py;  SLOTS=0,1,2,3;         STAGES="layers res" ;;
  *) echo "bad task"; exit 1 ;;
esac
R="$C/results/$SUITE"; LOGF="$R/driver.log"
log() { echo "[$(date '+%F %T')] $*" >> "$LOGF"; }
wait_pool() { while ps -eo args | grep -q "[g]pu_pool2.sh $1"; do sleep 120; done; }
qname() {                                                     # stage -> make_queue stage name
  case $TASK:$1 in ett:layers) echo tune_layers ;; ett:res) echo tune_res ;; mit:layers) echo mtune_layers ;; mit:res) echo mtune_res ;;
    mit:final) echo mfinal ;; cinc:layers) echo ctune_layers ;; cinc:res) echo ctune_res ;; esac; }
qfile() {                                                     # newest queue file of a stage, by its exact name prefix
  local pre; case $TASK in ett) pre="ett-$(qname $1)-" ;; *) pre="$(qname $1)-" ;; esac
  ls -t "$Q"/"$pre"*.txt 2>/dev/null | head -1; }
log "driver start ($TASK, suite $SUITE)"
prev=lr
for st in $STAGES; do
  # the pool of the current stage (S2 = layers was launched by the session) must finish first
  qf=$(qfile $st)
  if [ "$st" != layers ]; then                                 # build and launch this stage from the previous selection
    [ -s "$R/selected_$prev.json" ] || { log "STOP: selected_$prev.json missing"; exit 1; }
    (cd "$C/scripts" && bash make_queue_spk.sh "$(qname $st)") >> "$LOGF" 2>&1 || { log "STOP: make_queue $(qname $st) failed"; exit 1; }
    qf=$(qfile $st)
    if [ -s "$qf" ]; then
      rel=${qf#$N/}
      (cd "$N" && nohup bash scripts/gpu_pool2.sh "$rel" "$SLOTS" > "$R/pool_log_$st.txt" 2>&1 &)
      log "launched $(wc -l < "$qf") jobs: $rel"
      sleep 30
    else
      log "queue $qf empty"
    fi
  fi
  wait_pool "${qf#$N/}"
  log "pool done: ${qf#$N/} (RESULT $(grep -c '^RESULT' "$R/raw.txt"), SKIP $(grep -c '^SKIP' "$R/raw.txt"))"
  if [ "$TASK" = ett ]; then (cd "$C" && CUDA_VISIBLE_DEVICES=3 $PY energy_recount_ett.py --suite $SUITE --missing_only >> "$LOGF" 2>&1) || log "WARN: energy recount failed"; fi
  (cd "$C" && $PY $SUM --stage $st >> "$LOGF" 2>&1)
  [ -s "$R/selected_$st.json" ] || { log "STOP: no selection for $st (incomplete stage)"; exit 1; }
  log "selected_$st.json: $(tr -d '\n ' < "$R/selected_$st.json" | cut -c1-300)"
  prev=$st
done
if [ "$TASK" = mit ]; then                                     # final retraining of the tuned non-default settings on the whole file
  (cd "$C/scripts" && bash make_queue_spk.sh mfinal) >> "$LOGF" 2>&1 || { log "STOP: make_queue mfinal failed"; exit 1; }
  qf=$(qfile final)
  if [ -s "$qf" ]; then
    (cd "$N" && nohup bash scripts/gpu_pool2.sh "${qf#$N/}" "$SLOTS" > "$R/pool_log_final.txt" 2>&1 &)
    log "launched $(wc -l < "$qf") final jobs"; sleep 30; wait_pool "${qf#$N/}"; log "final pool done"
  fi
  (cd "$C" && CUDA_VISIBLE_DEVICES=3 $PY energy_recount_mitbih.py --suite $SUITE >> "$LOGF" 2>&1) || log "WARN: energy recount failed"
fi
(cd "$C" && $PY $SUM --stage final >> "$LOGF" 2>&1) && log "final summary written: $R/summary_final.md" || log "STOP: final summary failed"
log "driver end"
