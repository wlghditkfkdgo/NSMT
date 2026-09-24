#!/usr/bin/env bash
# Prereg 2J, D-AM: selection stage for mode=hard on validation, seed 7.
#   q in {1.0 (baseline == full, gate G18), 0.10, 0.25, 0.50}, shared axis, Pearson.
# Exploratory training settings, identical to the eta grid (2A-2I): 2048 train / 256 val
# sequences, 12 epochs, batch 64, CPU, calibrated input_scale/theta/G11 bound.
# The test split is NOT evaluated (--no-test): 2J uses val for selection and confirm2 for the
# one confirmatory look. Runs go in parallel; each gets its own log under the suite.
#
#   bash scripts/hard_select.sh <suite>        e.g. hardsel-$(date +%H%M%S)
set -euo pipefail
cd "$(dirname "$0")/.."
SUITE=${1:?suite name}
export LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
mkdir -p "log/$SUITE"
for Q in 1 0.1 0.25 0.5; do
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=${THREADS:-4} MKL_NUM_THREADS=${THREADS:-4} \
  nohup $PY train.py --task recall --data recall --suite "$SUITE" --mode hard \
      --hard_axis shared --hard_stat pearson --hard_q "$Q" --seed 7 \
      --n_train 2048 --n_val 256 --n_test 256 -e 12 -bs 64 --cpu --no-test \
      > "log/$SUITE/train_q$Q.log" 2>&1 &
  echo "launched q=$Q pid $!"
done
wait
echo "all four finished"
