#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
mkdir -p "$ROOT/log"
source /home/anaconda3/etc/profile.d/conda.sh; conda activate snn_jelly
export MKL_THREADING_LAYER=GNU MKL_SERVICE_FORCE_INTEL=0
GPUS=(0 1 2); DATASETS=(MSL SMD SMAP PSM SWAT)
worker(){ local gpu=$1 wid=$2 k=0 ds
  for ds in "${DATASETS[@]}"; do
    if (( k % 3 == wid )); then
      echo "[optuna-active] $ds -> gpu$gpu $(date +%H:%M)"
      python3 hp_search_optuna.py --data "$ds" --gpu "$gpu" --n_trials 20 --epochs 15 > "$ROOT/log/hp_active_${ds}.out" 2>&1
    fi; k=$((k+1))
  done; }
for w in 0 1 2; do worker "${GPUS[$w]}" "$w" & done
wait; echo "DONE hp_active"
