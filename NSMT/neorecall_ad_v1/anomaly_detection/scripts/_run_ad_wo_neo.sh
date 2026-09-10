#!/usr/bin/env bash
# Causal test of Memory Replay on Anomaly Detection: train the FULL myModel and
# the w/o-Neo ablation (Memory Replay receives zero memory) under IDENTICAL
# Optuna-best HP / seed / epochs, on the fast datasets (PSM, MSL, SMAP). Compare
# adjusted F1. If wo_neo ~= full, the memory is causally inert on AD too.
# One dataset per GPU (0,1,2); full then wo_neo sequentially.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="log/ad_wo_neo"
SUMMARY_DIR="$ROOT/$LOG_DIR/launcher"
mkdir -p "$SUMMARY_DIR"
cd "$ROOT"

source /home/anaconda3/etc/profile.d/conda.sh
conda activate snn_jelly

# fields: emb nh patch_size lr alpha time_layers mlp_ratios keep_ratio c_out bs anomaly_ratio root
declare -A CONF
CONF[MSL]="32 16 8  5.5842e-05 0.328 1 2 0.0  55 64  1.0 MSL"
CONF[SMAP]="32 4 12 0.00042357 0.196 1 2 0.1  25 32  1.0 SMAP"
CONF[PSM]="32 4 8  0.00086274 0.400 4 6 0.25 25 128 1.0 PSM"
declare -A GPU
GPU[PSM]=0; GPU[MSL]=1; GPU[SMAP]=2

run_one() {  # model_key data
    local mk="$1" data="$2"; local gpu="${GPU[$data]}"
    IFS=' ' read -r emb nh ps lr al tl mr kr c_out bs ar rootname <<< "${CONF[$data]}"
    local logf="$SUMMARY_DIR/${data}_${mk}.log"
    echo "[run] $mk $data on GPU=$gpu" | tee "$logf"
    python3 ./train.py --model "$mk" --gating attn --no-bias --scheduler reduce -s --test \
        -nd "$gpu" -e 30 --warm_up_epoch 0 \
        -bs "$bs" -emb "$emb" -nh "$nh" -lr "$lr" \
        --alpha "$al" --time_layers "$tl" --keep_ratio "$kr" --mlp_ratios "$mr" \
        --data "$data" --pred_len 0 --patch_size "$ps" \
        --root_path "$ROOT/dataset/$rootname" --seq_len 100 --c_out "$c_out" \
        --anomaly_ratio "$ar" --features M \
        --log_dir "$LOG_DIR" --tag "$mk" \
        >> "$logf" 2>&1
}

run_dataset() { run_one myModel "$1"; run_one wo_neo "$1"; }

for data in PSM MSL SMAP; do run_dataset "$data" & done
wait
echo "DONE AD full-vs-wo_neo sweep."
