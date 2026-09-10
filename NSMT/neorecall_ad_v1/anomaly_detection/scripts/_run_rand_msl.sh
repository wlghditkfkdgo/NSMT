#!/usr/bin/env bash
# Add rand (random past-token selection) to the MSL anomaly-detection validation, to compare
# vs existing none/sine (sine_msl: none 0.7427, sine 0.7445). 3 seeds. F1 (point-adjusted).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
DATA_ROOT="$ROOT/dataset/MSL"
SUMMARY_DIR="$ROOT/log/sine_msl/launcher"; mkdir -p "$SUMMARY_DIR"
source /home/anaconda3/etc/profile.d/conda.sh; conda activate snn_jelly
GPUS=(3 4 5)
i=0
for seed in 42 7 2026; do
  gpu=${GPUS[$(( i % 3 ))]}; i=$((i+1))
  logf="$SUMMARY_DIR/MSL_rand_s${seed}.log"; : > "$logf"
  echo "[run] MSL rand seed=$seed GPU=$gpu" | tee -a "$logf"
  python3 ./train.py --model myModel --gating attn --no-bias --scheduler reduce -s --test \
    --seed "$seed" -nd "$gpu" -e 100 --warm_up_epoch 0 -bs 64 -emb 32 -nh 16 -lr 5.5842e-05 \
    --keep_ratio 0.25 --alpha 0.3279 --time_layers 1 --mlp_ratios 2 --data MSL --pred_len 0 \
    --patch_size 12 --roll_mode rand --root_path "$DATA_ROOT" --seq_len 100 --c_out 55 \
    --anomaly_ratio 1 --features M >> "$logf" 2>&1 &
done
wait
echo "DONE MSL rand."
