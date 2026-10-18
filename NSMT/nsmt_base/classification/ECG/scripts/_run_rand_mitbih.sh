#!/usr/bin/env bash
# Add rand (random past-token selection) to MIT-BIH ECG validation, vs existing none/sine
# (../results/ECG/sine_mitbih: none 0.9143, sine 0.9150). 3 seeds. macro-F1.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
DATA_ROOT="$ROOT/data"
SUMMARY_DIR="$ROOT/../results/ECG/sine_mitbih/launcher"; mkdir -p "$SUMMARY_DIR"
source /home/anaconda3/etc/profile.d/conda.sh; conda activate snn_jelly
GPUS=(3 4 5)
i=0
for seed in 42 7 2026; do
  gpu=${GPUS[$(( i % 3 ))]}; i=$((i+1))
  logf="$SUMMARY_DIR/mitbih_rand_s${seed}.log"; : > "$logf"
  echo "[run] mitbih rand seed=$seed GPU=$gpu" | tee -a "$logf"
  python3 train.py --model myModel --seed "$seed" --log_dir ../results/ECG/sine_mitbih --gating attn --attn SSA \
    -s --test --print_epoch 1 -nd "$gpu" -e 30 --warm_up_epoch 0 --num_channels 1 -bs 512 -lr 0.01 \
    --max_lr 0.005 -emb 384 -nh 8 --mlp_ratios 1 --layers 1 --keep_ratio 0.15 --time_layers 2 \
    --alpha 0.1 --patch_size 32 --no-bias --scheduler cosine --roll_mode rand \
    --data UEA --data_path ECG --seq_len 187 --root_path "$DATA_ROOT" >> "$logf" 2>&1 &
done
wait
echo "DONE mitbih rand."
