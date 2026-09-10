#!/usr/bin/env bash
# Launch per-dataset Optuna HP search in parallel (one GPU per dataset).
#
# Usage:
#   bash scripts/_run_hp_optuna.sh [N_TRIALS]
# Default N_TRIALS = 20 per dataset.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUMMARY_DIR="$ROOT/log/hp_optuna/launcher"
mkdir -p "$SUMMARY_DIR"
cd "$ROOT"

source /home/anaconda3/etc/profile.d/conda.sh
conda activate snn_jelly

N_TRIALS="${1:-20}"

# dataset -> gpu mapping
declare -A GPU
GPU[MSL]=1
GPU[SMAP]=2
GPU[SMD]=3
GPU[PSM]=4
GPU[SWAT]=5

for data in MSL SMAP SMD PSM SWAT; do
    gpu="${GPU[$data]}"
    logf="$SUMMARY_DIR/${data}_optuna.log"
    echo "[launch] $data -> gpu=$gpu n_trials=$N_TRIALS log=$logf"
    python3 ./hp_search_optuna.py \
        --data "$data" --gpu "$gpu" --n_trials "$N_TRIALS" \
        --epochs 15 --patience 2 \
        > "$logf" 2>&1 &
done
wait

echo "DONE Optuna HP search across all AD datasets."
echo "Best configs: $ROOT/log/hp_optuna/<DATASET>/best.json"
