#!/usr/bin/env bash
# Validate the sine spike-encoding (raised-cosine causal shift, APE-like) on anomaly detection.
# roll_mode none(original repeat) vs sine, MSL (lightest AD dataset), 3 seeds. Metric: F1 (higher better).
# Focus: is sine effective on a DIFFERENT task than forecasting? NOTE: need not help every dataset.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
DATA_ROOT="$ROOT/dataset/MSL"
SUMMARY_DIR="$ROOT/log/sine_msl/launcher"; mkdir -p "$SUMMARY_DIR"
source /home/anaconda3/etc/profile.d/conda.sh; conda activate snn_jelly
run_one() { # seed gpu mode
  local seed="$1" gpu="$2" mode="$3"
  local logf="$SUMMARY_DIR/MSL_${mode}_s${seed}.log"; : > "$logf"
  echo "[run] MSL $mode seed=$seed GPU=$gpu" | tee -a "$logf"
  python3 ./train.py --model myModel --gating attn --no-bias --scheduler reduce -s --test \
    --seed "$seed" -nd "$gpu" -e 100 --warm_up_epoch 0 -bs 64 -emb 32 -nh 16 -lr 5.5842e-05 \
    --keep_ratio 0.25 --alpha 0.3279 --time_layers 1 --mlp_ratios 2 --data MSL --pred_len 0 \
    --patch_size 12 --roll_mode "$mode" --root_path "$DATA_ROOT" --seq_len 100 --c_out 55 \
    --anomaly_ratio 1 --features M >> "$logf" 2>&1
}
i=0
for mode in none sine; do for seed in 42 7 2026; do
  gpu=$(( i % 6 )); run_one "$seed" "$gpu" "$mode" & i=$((i+1))
done; done
wait
echo "DONE sine_msl sweep."
