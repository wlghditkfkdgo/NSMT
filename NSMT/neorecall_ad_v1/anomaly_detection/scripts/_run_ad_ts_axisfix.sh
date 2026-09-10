#!/usr/bin/env bash
# AD ts with the corrected student axis-fix (Neocortex output [N(=T),BC,1,D]
# -> token axis [1,BC,N,D] before the joint neo_head, instead of replicating a
# single token N times). Fast datasets only (PSM, MSL, SMAP) for a quick read,
# one GPU each. Same Optuna-best HPs as _run_ad_ts.sh so the only difference is
# the student axis handling. Compare adj-F1 vs ad_ts/ (axis-bug) and baseline.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="log/ad_ts_axisfix"
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

run_dataset() {
    local data="$1"; local gpu="${GPU[$data]}"
    IFS=' ' read -r emb nh ps lr al tl mr kr c_out bs ar rootname <<< "${CONF[$data]}"
    local logf="$SUMMARY_DIR/${data}.log"
    echo "[run] ts-axisfix $data on GPU=$gpu (emb=$emb nh=$nh ps=$ps tl=$tl)" | tee "$logf"
    python3 ./train.py --model ts --gating attn --no-bias --scheduler reduce -s --test \
        -nd "$gpu" -e 30 --warm_up_epoch 0 \
        -bs "$bs" -emb "$emb" -nh "$nh" -lr "$lr" \
        --alpha "$al" --time_layers "$tl" --keep_ratio "$kr" --mlp_ratios "$mr" \
        --data "$data" --pred_len 0 --patch_size "$ps" \
        --root_path "$ROOT/dataset/$rootname" --seq_len 100 --c_out "$c_out" \
        --anomaly_ratio "$ar" --features M \
        --log_dir "$LOG_DIR" --tag "ts_axisfix" \
        >> "$logf" 2>&1
}

for data in PSM MSL SMAP; do run_dataset "$data" & done
wait
echo "DONE AD ts-axisfix sweep."
