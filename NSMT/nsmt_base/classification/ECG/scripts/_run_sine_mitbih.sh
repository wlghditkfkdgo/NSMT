#!/usr/bin/env bash
# Validate the sine spike-encoding (raised-cosine causal shift, APE-like) on MIT-BIH ECG
# classification (5-class, imbalanced). roll_mode none(repeat) vs sine, model myModel, 3 seeds.
# Metric: macro-F1 (key for imbalanced classes). Focus: is sine effective on classification?
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
DATA_ROOT="$ROOT/data"
SUMMARY_DIR="$ROOT/../results/ECG/sine_mitbih/launcher"; mkdir -p "$SUMMARY_DIR"
source /home/anaconda3/etc/profile.d/conda.sh; conda activate snn_jelly
run_one() { # seed gpu mode
  local seed="$1" gpu="$2" mode="$3"
  local logf="$SUMMARY_DIR/mitbih_${mode}_s${seed}.log"; : > "$logf"
  echo "[run] mitbih $mode seed=$seed GPU=$gpu" | tee -a "$logf"
  python3 train.py --model myModel --seed "$seed" --log_dir ../results/ECG/sine_mitbih --gating attn --attn SSA \
    -s --test --print_epoch 1 -nd "$gpu" -e 30 --warm_up_epoch 0 --num_channels 1 -bs 512 -lr 0.01 \
    --max_lr 0.005 -emb 384 -nh 8 --mlp_ratios 1 --layers 1 --keep_ratio 0.15 --time_layers 2 \
    --alpha 0.1 --patch_size 32 --no-bias --scheduler cosine --roll_mode "$mode" \
    --data UEA --data_path ECG --seq_len 187 --root_path "$DATA_ROOT" >> "$logf" 2>&1
}
i=0
for mode in none sine; do for seed in 42 7 2026; do
  gpu=$(( i % 6 )); run_one "$seed" "$gpu" "$mode" & i=$((i+1))
done; done
wait
echo "DONE ../results/ECG/sine_mitbih sweep."
