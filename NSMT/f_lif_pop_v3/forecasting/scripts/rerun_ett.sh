#!/usr/bin/env bash
# Prereg 2T: one ETT training run of the 2N / 2O / 2P / 2Q (B) rerun with the spikeDE-based neurons.
# The protocol is run_ett.sh's (left as it was run): AdamW lr 1e-3 wd 1e-2, batch 128, clip 1,
# max 50 epochs, early stop 10 on val MSE, ReduceLROnPlateau(0.5, 5), CUDA, --no-test. Only the
# neuron changes:
#   q1*      = population f-LIF + soma, no selection        (--neuron pop_flif --soma lif)
#   pearson* = the same + Selective, pearson, shared, q 0.5  (--neuron sel_pop_flif --soma lif)
#   _R = window normalisation R (--revin), _a1 = alpha 1, _lin = one nn.Linear readout (2Q B).
# The baselines (GRU, Linear) do not depend on the neuron and are not retrained (2T).
# usage: SUITE=<suite> rerun_ett.sh <DS> <PL> <cond> <seed> <gpu>
set -u
DS=${1:?}; PL=${2:?}; C=${3:?}; S=${4:?}; GPU=${5:?}
SUITE=${SUITE:?set SUITE}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # task root
R="$(cd "$W/../.." && pwd)"  # NSMT root
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}   # torch 1.12.0+cu113: the declared topk tie rule
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EPOCHS=${EPOCHS:-50}                                       # override to 1 for smoke tests
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}
mkdir -p "$RES" "$LOG"
RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
Q1="--model myModel --neuron pop_flif --soma lif --mode full"
SEL="--model myModel --neuron sel_pop_flif --soma lif --mode hard --hard_axis shared --hard_stat pearson --hard_q 0.5"
case $C in
  q1)            MF="$Q1" ;;                                   # 2N
  pearson)       MF="$SEL" ;;
  q1_R)          MF="$Q1 --revin" ;;                           # 2O
  pearson_R)     MF="$SEL --revin" ;;
  q1_R_a1)       MF="$Q1 --revin --alpha 1" ;;                 # 2P
  pearson_R_a1)  MF="$SEL --revin --alpha 1" ;;
  q1_R_lin)      MF="$Q1 --revin --head_mode linear" ;;        # 2Q (B)
  pearson_R_lin) MF="$SEL --revin --head_mode linear" ;;
  *) echo "bad cond $C"; exit 1 ;;
esac
IX="ett_${DS}_p${PL}_${C}"; OUT=$LOG/${IX}_seed${S}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train.py --task ett --data $DS \
  --pred_len $PL --suite $SUITE --seed $S -nd 0 -e $EPOCHS --no-test $MF ${EXTRA:-} > "$OUT" 2>&1)
RC=$?
VAL=$(grep -aoE 'Validation loss decreased \([^)]*--> [0-9.]+' "$OUT" | tail -1 | grep -oE '[0-9.]+$')
PAR=$(grep -a '\[train\] parameters' "$OUT" | grep -oE '[0-9]+' | head -1); t1=$(date +%s)
EP=$(grep -ac '^epoch ' "$OUT")
if [ $RC -ne 0 ] || [ -z "$VAL" ]; then
  G11=$(grep -ac 'G11 violated' "$OUT")
  echo "SKIP(rc=$RC, no val, g11=$G11) ds=$DS pl=$PL cond=$C seed=$S time_s=$((t1-t0)) sha=$SHA" | tee -a $RES/raw.txt; exit 1
fi
echo "RESULT exp=ett dataset=${DS}_p${PL} cond=${C} seed=$S best_val_mse=$VAL epochs=$EP params=$PAR time_s=$((t1-t0)) gpu=$GPU sha=$SHA" | tee -a $RES/raw.txt
