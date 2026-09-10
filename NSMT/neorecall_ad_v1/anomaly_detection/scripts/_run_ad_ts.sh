#!/usr/bin/env bash
# #4: Teacher->Student (--model ts) on the AD task, all 5 datasets, using the
# SAME Optuna-best HPs as the baseline so the only difference is the
# architecture. Single GPU (budget). Anomaly score uses the Student
# (Neocortex) reconstruction; anomaly_ratio is swept at eval.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="log/ad_ts"
SUMMARY_DIR="$ROOT/$LOG_DIR/launcher"
mkdir -p "$SUMMARY_DIR"
cd "$ROOT"

source /home/anaconda3/etc/profile.d/conda.sh
conda activate snn_jelly

GPU="${1:-2}"

# fields: emb nh patch_size lr alpha time_layers mlp_ratios keep_ratio c_out bs anomaly_ratio root
declare -A CONF
CONF[MSL]="32 16 8  5.5842e-05 0.328 1 2 0.0  55 64  1.0 MSL"
CONF[SMAP]="32 4 12 0.00042357 0.196 1 2 0.1  25 32  1.0 SMAP"
CONF[SMD]="32 16 8  0.00086999 0.140 2 4 0.0  38 128 0.5 SMD"
CONF[PSM]="32 4 8  0.00086274 0.400 4 6 0.25 25 128 1.0 PSM"
CONF[SWAT]="32 4 12 0.00086274 0.400 2 2 0.0  51 32  1.0 SWaT"

for data in SMD PSM MSL SMAP SWAT; do
    IFS=' ' read -r emb nh ps lr al tl mr kr c_out bs ar rootname <<< "${CONF[$data]}"
    logf="$SUMMARY_DIR/${data}.log"
    echo "[run] ts $data on GPU=$GPU (emb=$emb nh=$nh ps=$ps tl=$tl)" | tee "$logf"
    python3 ./train.py --model ts --gating attn --no-bias --scheduler reduce -s --test \
        -nd "$GPU" -e 30 --warm_up_epoch 0 \
        -bs "$bs" -emb "$emb" -nh "$nh" -lr "$lr" \
        --alpha "$al" --time_layers "$tl" --keep_ratio "$kr" --mlp_ratios "$mr" \
        --data "$data" --pred_len 0 --patch_size "$ps" \
        --root_path "$ROOT/dataset/$rootname" --seq_len 100 --c_out "$c_out" \
        --anomaly_ratio "$ar" --features M \
        --log_dir "$LOG_DIR" --tag "ts" \
        >> "$logf" 2>&1
done

echo "DONE AD ts sweep."
