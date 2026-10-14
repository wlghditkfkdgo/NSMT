#!/usr/bin/env bash
# Pipeline smoke test of the neurons.py classes (--neuron): D11 calibration, then a short training
# run, for one class configuration. NOT an experiment: 2 epochs on 20 train batches, validation on
# 5 batches, no test split (--no-test). Calibrations go to their own folder (--calibration_dir) so
# the real results/calibration stays untouched; results and logs go to the smoke suite.
# usage: neurons_smoke.sh <cond> <gpu>     (EXTRA="..." adds training options, e.g. an exploratory run)
set -u
C=${1:?}; GPU=${2:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # task root
R="$(cd "$W/../.." && pwd)"  # NSMT root
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}   # torch 1.12.0+cu113
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EPOCHS=${EPOCHS:-2}
SUITE=${SUITE:-neurons-smoke-20261014}
CAL=${CAL:-calibration-$SUITE}
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}
mkdir -p "$RES" "$LOG"
RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
ONE="--tau 8 --num_population 1"                           # 한 뉴런 클래스: tau 하나
case $C in
  lif)                MF="--neuron lif --alpha 1 $ONE --mode full" ;;
  flif)               MF="--neuron flif $ONE --mode full" ;;
  sel_flif)           MF="--neuron sel_flif $ONE --mode hard --hard_axis shared --hard_stat pearson --hard_q 0.5" ;;
  pop_flif)           MF="--neuron pop_flif --mode full" ;;
  pop_flif_none)      MF="--neuron pop_flif --soma none --mode full" ;;
  sel_pop_flif)       MF="--neuron sel_pop_flif --mode hard --hard_axis shared --hard_stat pearson --hard_q 0.5" ;;
  sel_pop_flif_input) MF="--neuron sel_pop_flif --mode hard --hard_axis shared --hard_stat pearson --hard_q 0.5 --select_part input" ;;
  *) echo "bad cond $C"; exit 1 ;;
esac
COMMON="--task ett --data ETTh1 --pred_len 96 --revin --seed 7 -nd 0 --calibration_dir $CAL"
IX="smoke_${C}_seed7_${TS}_${SHA}"; t0=$(date +%s)
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./calibrate.py $COMMON $MF \
  > "$LOG/${IX}_calibrate.log" 2>&1)
RC_CAL=$?
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train.py $COMMON $MF --suite $SUITE \
  -e $EPOCHS --max_train_batches 20 --max_eval_batches 5 --no-test ${EXTRA:-} > "$LOG/${IX}_train.log" 2>&1)
RC=$?
PICK=$(grep -a 'picked input_scale' "$LOG/${IX}_calibrate.log" | grep -oE 'input_scale = [0-9.]+' | grep -oE '[0-9.]+$')
VAL=$(grep -aoE 'Validation loss decreased \([^)]*--> [0-9.]+' "$LOG/${IX}_train.log" | tail -1 | grep -oE '[0-9.]+$')
PAR=$(grep -a '\[train\] parameters' "$LOG/${IX}_train.log" | grep -oE '[0-9]+' | head -1); t1=$(date +%s)
echo "SMOKE cond=$C calibrate_rc=$RC_CAL input_scale=${PICK:-none} train_rc=$RC best_val_mse=${VAL:-none} params=${PAR:-none} time_s=$((t1-t0)) sha=$SHA" | tee -a "$RES/raw.txt"
[ $RC_CAL -eq 0 ] && [ $RC -eq 0 ] && [ -n "$VAL" ]
