#!/usr/bin/env bash
# Prereg 2K, D-AQ: same-budget controls. Eight pre-registered seeds x {recent, random} at
# q = 0.5, the 2J settings. The pearson and q=1 models come from the 2J confirmation suite
# (hardconf-015047) and are NOT retrained. Test and confirm3 are not touched here.
#
#   bash scripts/hard_control_train.sh <suite>      e.g. hardctrl-$(date +%H%M%S)
set -euo pipefail
cd "$(dirname "$0")/.."
SUITE=${1:?suite name}
SEEDS=(7 13 21 42 123 256 512 1024)
export LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
mkdir -p "log/$SUITE"
for SEED in "${SEEDS[@]}"; do
  for STAT in recent random; do
    CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=${THREADS:-4} MKL_NUM_THREADS=${THREADS:-4} \
    nohup $PY train.py --task recall --data recall --suite "$SUITE" --mode hard \
        --hard_axis shared --hard_stat "$STAT" --hard_q 0.5 --seed "$SEED" \
        --n_train 2048 --n_val 256 --n_test 256 -e 12 -bs 64 --cpu --no-test \
        > "log/$SUITE/train_seed${SEED}_$STAT.log" 2>&1 &
    echo "launched seed=$SEED stat=$STAT pid $!"
  done
done
wait
echo "all sixteen finished"
