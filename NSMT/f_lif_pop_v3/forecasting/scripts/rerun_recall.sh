#!/usr/bin/env bash
# Prereg 2T: recall-task training for the 2J / 2K rerun with the spikeDE-based neurons. The settings
# are hard_select.sh / hard_confirm_train.sh / hard_control_train.sh's (left as they were run):
# 2048 train / 256 val / 256 test sequences, 12 epochs, batch 64, CPU, --no-test, calibrated
# input_scale and G11 bound. Only the neuron changes:
#   q = 1   -> --neuron pop_flif --soma lif --mode full          (the plain population f-LIF)
#   q < 1   -> --neuron sel_pop_flif --soma lif --mode hard --hard_axis shared --hard_stat <stat>
# One seed builds one initial model for every condition (check_neurons N18). No split beyond val
# is evaluated here; confirm2 / confirm3 / confirm4 are opened later by analysis/recall_rerun.py.
#   select  <suite>          seed 7 x q {1, 0.1, 0.25, 0.5} (pearson)               4 runs
#   confirm <suite> <q>      8 seeds x {q = 1, pearson q}                          16 runs
#   control <suite> <q>      8 seeds x {recent, random} at q                       16 runs
# usage: rerun_recall.sh <select|confirm|control> <suite> [q]      (runs in parallel, then waits)
set -u
STAGE=${1:?select|confirm|control}; SUITE=${2:?suite}; QSEL=${3:-}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # task root
R="$(cd "$W/../.." && pwd)"  # NSMT root
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EPOCHS=${EPOCHS:-12}                                       # override to 1 for smoke tests
RES=$W/results/$SUITE; LOG=$W/log/$SUITE; mkdir -p "$RES" "$LOG"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
SEEDS=(7 13 21 42 123 256 512 1024)
COMMON="--task recall --data recall --suite $SUITE --n_train 2048 --n_val 256 --n_test 256 -e $EPOCHS -bs 64 --cpu --no-test"

flags() {   # <stat> <q>
  if [ "$2" = 1 ]; then echo "--neuron pop_flif --soma lif --mode full"
  else echo "--neuron sel_pop_flif --soma lif --mode hard --hard_axis shared --hard_stat $1 --hard_q $2"; fi
}

one() {     # <seed> <stat> <q>   one run, then a RESULT / SKIP line
  local s=$1 st=$2 q=$3 label
  label=$([ "$q" = 1 ] && echo q1 || echo "${st}_q$q")
  local out=$LOG/recall_${label}_seed${s}_${TS}_${SHA}.stdout t0=$(date +%s)
  (cd "$W" && CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=${THREADS:-2} MKL_NUM_THREADS=${THREADS:-2} \
     $PY ./train.py $COMMON --seed $s $(flags $st $q) ${EXTRA:-} > "$out" 2>&1)
  local rc=$? t1=$(date +%s)
  local val=$(grep -aoE 'Validation loss decreased \([^)]*--> [0-9.]+' "$out" | tail -1 | grep -oE '[0-9.]+$')
  local ep=$(grep -ac '^epoch ' "$out")
  if [ $rc -ne 0 ] || [ -z "$val" ]; then
    echo "SKIP(rc=$rc, no val, g11=$(grep -ac 'G11 violated' "$out")) exp=recall cond=$label seed=$s time_s=$((t1-t0)) sha=$SHA" >> "$RES/raw.txt"
  else
    echo "RESULT exp=recall cond=$label seed=$s best_val_loss=$val epochs=$ep time_s=$((t1-t0)) sha=$SHA" >> "$RES/raw.txt"
  fi
}

case $STAGE in
  select)  for Q in 1 0.1 0.25 0.5; do one 7 pearson $Q & done ;;
  confirm) [ -n "$QSEL" ] || { echo "confirm needs the selected q"; exit 1; }
           for S in "${SEEDS[@]}"; do one $S pearson 1 & one $S pearson $QSEL & done ;;
  control) [ -n "$QSEL" ] || { echo "control needs the selected q"; exit 1; }
           for S in "${SEEDS[@]}"; do one $S recent $QSEL & one $S random $QSEL & done ;;
  *) echo "bad stage $STAGE"; exit 1 ;;
esac
wait
echo "[rerun_recall] $STAGE finished: $(grep -c '^RESULT' "$RES/raw.txt") RESULT / $(grep -c '^SKIP' "$RES/raw.txt") SKIP lines in $RES/raw.txt"
