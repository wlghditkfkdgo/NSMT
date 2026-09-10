#!/usr/bin/env bash
# AD HP search.
#
# We train each dataset ONCE with the existing per-dataset baseline config,
# then run anomaly_detection/sweep_ar.py to evaluate multiple anomaly_ratio
# values without retraining (anomaly_ratio only changes the threshold).
#
# Each dataset gets its own GPU. SMAP / SWaT use a smaller batch size to fit
# their larger emb=128 models in memory.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="log/hp_search_v2"
SUMMARY_DIR="$ROOT/$LOG_DIR/launcher"
mkdir -p "$SUMMARY_DIR"
cd "$ROOT"

source /home/anaconda3/etc/profile.d/conda.sh
conda activate snn_jelly

# fields: gpu emb nh patch_size lr alpha time_layers mlp_ratios keep_ratio c_out seq_len bs epoch
declare -A BASE
BASE[MSL]="1 64  8 12  0.0005 0.5 4 4 0    55 100 128 30"
BASE[SMAP]="2 128 8 8   0.0005 0.1 2 2 0.1  25 100 32  30"
BASE[SMD]="3 32  8 8   0.0005 0.2 2 4 0    38 100 128 30"
BASE[PSM]="4 64  8 16  0.0001 0.9 4 4 0.25 25 100 128 30"
BASE[SWAT]="5 128 8 12  0.0005 0.8 1 6 0    51 100 32  30"

declare -A ROOTNAME
ROOTNAME[MSL]=MSL
ROOTNAME[SMAP]=SMAP
ROOTNAME[SMD]=SMD
ROOTNAME[PSM]=PSM
ROOTNAME[SWAT]=SWaT     # on-disk folder uses lowercase 'a'

train_then_sweep () {
    local data="$1"
    IFS=' ' read -r gpu emb nh ps lr al tl mr kr c_out sl bs ep <<< "${BASE[$data]}"
    local rname="${ROOTNAME[$data]}"
    local logf="$SUMMARY_DIR/${data}.log"

    echo "[launch] $data on GPU $gpu (bs=$bs, e=$ep)"
    python3 ./train.py \
        --model myModel --gating attn --no-bias --scheduler reduce -s --test \
        -nd "$gpu" -e "$ep" --warm_up_epoch 0 \
        -bs "$bs" -emb "$emb" -nh "$nh" -lr "$lr" \
        --alpha "$al" --time_layers "$tl" --keep_ratio "$kr" --mlp_ratios "$mr" \
        --data "$data" --pred_len 0 \
        --patch_size "$ps" --root_path "$ROOT/dataset/$rname" \
        --seq_len "$sl" --c_out "$c_out" \
        --anomaly_ratio 1.0 --features M \
        --log_dir "$LOG_DIR" --tag "phaseA" \
        > "$logf" 2>&1

    # Find the run directory and re-evaluate over the AR grid.
    local rdir
    rdir=$(grep -oE "Final result saved to \`[^\`]+\`" "$logf" | tail -1 | sed -E 's/^.*`([^`]+)`.*$/\1/')
    if [ -z "$rdir" ]; then
        echo "[$data] training did not finish; skipping AR sweep"
        return
    fi
    # rdir is .../log; the saved config sits at ../model_state/config.pt
    local run_root
    run_root=$(dirname "$rdir")
    echo "[sweep] $data run_root=$run_root"
    python3 ./sweep_ar.py \
        --config "$run_root" --model myModel -nd "$gpu" \
        --ar_grid 0.3 0.5 0.75 1.0 1.5 2.0 2.5 3.0 \
        >> "$logf" 2>&1
}

for data in MSL SMAP SMD PSM SWAT; do
    train_then_sweep "$data" &
done
wait

echo "DONE AD HP search v2"
echo "Sweep CSVs are at <run>/log/anomaly_ratio_sweep.csv"
