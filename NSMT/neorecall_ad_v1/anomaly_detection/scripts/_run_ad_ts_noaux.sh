#!/usr/bin/env bash
# AD ts variant with the DCT low-frequency AUXILIARY loss DISABLED (--alpha 0).
# Everything else identical to scripts/_run_ad_ts.sh (same Optuna-best HP,
# same architecture). The KD loss (student<->teacher + reconstruct input) stays.
# Compare adjusted F1 against the aux-on ts results (ad_ts/).
#
# 4 datasets covering best/medium/worst aux-on F1, one GPU each (budget <=4):
#   PSM (best 0.959), SMD (0.839), SWAT (0.882), SMAP (worst 0.675)

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="log/ad_ts_noaux"
SUMMARY_DIR="$ROOT/$LOG_DIR/launcher"
mkdir -p "$SUMMARY_DIR"
cd "$ROOT"

source /home/anaconda3/etc/profile.d/conda.sh
conda activate snn_jelly

# fields: gpu emb nh patch_size lr time_layers mlp_ratios keep_ratio c_out bs anomaly_ratio rootname
declare -A CONF
CONF[PSM]="0  32 4 8  0.00086274 4 6 0.25 25 128 1.0 PSM"
CONF[SMD]="1  32 16 8 0.00086999 2 4 0.0  38 128 0.5 SMD"
CONF[SWAT]="2 32 4 12 0.00086274 2 2 0.0  51 32  1.0 SWaT"
CONF[SMAP]="3 32 4 12 0.00042357 1 2 0.1  25 32  1.0 SMAP"

run_one() {  # data
    local data="$1"
    IFS=' ' read -r gpu emb nh ps lr tl mr kr c_out bs ar rootname <<< "${CONF[$data]}"
    local logf="$SUMMARY_DIR/${data}.log"
    echo "[run] ts(aux-off, alpha=0) $data on GPU=$gpu" | tee "$logf"
    python3 ./train.py --model ts --gating attn --no-bias --scheduler reduce -s --test \
        -nd "$gpu" -e 30 --warm_up_epoch 0 \
        -bs "$bs" -emb "$emb" -nh "$nh" -lr "$lr" \
        --alpha 0 --time_layers "$tl" --keep_ratio "$kr" --mlp_ratios "$mr" \
        --data "$data" --pred_len 0 --patch_size "$ps" \
        --root_path "$ROOT/dataset/$rootname" --seq_len 100 --c_out "$c_out" \
        --anomaly_ratio "$ar" --features M \
        --log_dir "$LOG_DIR" --tag "ts_noaux" \
        >> "$logf" 2>&1
}

for data in PSM SMD SWAT SMAP; do
    run_one "$data" &
done
wait

echo "DONE AD ts aux-off sweep."
