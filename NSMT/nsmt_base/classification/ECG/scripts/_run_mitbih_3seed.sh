#!/usr/bin/env bash
# MIT-BIH clean 3-seed: {no-Neo, passive, active} x 3 seeds, distinct log_dirs.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source /home/anaconda3/etc/profile.d/conda.sh; conda activate snn_jelly
declare -A EX; EX[noneo]="--no_neo"; EX[passive]=""; EX[active]="--neo_recurrent --neo_tau 8"
GPUS=(0 1 2); i=0; NG=${#GPUS[@]}
for cond in noneo passive active; do
  for seed in 42 7 2026; do
    gpu=${GPUS[$(( i % NG ))]}; i=$((i+1))
    ld="../results/ECG/mb3_${cond}"; sd="$ROOT/$ld/launcher"; mkdir -p "$sd"; logf="$sd/s${seed}.log"
    grep -q "f1\|adj" "$logf" 2>/dev/null && { echo "[skip] $cond s$seed"; continue; }
    : > "$logf"; echo "[run] $cond s$seed GPU=$gpu" | tee -a "$logf"
    nohup python3 ./train.py --model myModel --seed "$seed" --log_dir "$ld" --gating attn --attn SSA -s --test \
      --print_epoch 1 -nd "$gpu" -e 30 --warm_up_epoch 0 --num_channels 1 -bs 512 -lr 0.01 --max_lr 0.005 \
      -emb 384 -nh 8 --mlp_ratios 1 --layers 1 --keep_ratio 0.15 --time_layers 2 --alpha 0.1 --patch_size 32 \
      --no-bias --scheduler cosine ${EX[$cond]} --data UEA --data_path ECG --seq_len 187 \
      --root_path "$ROOT/data" >> "$logf" 2>&1 &
    (( i % NG == 0 )) && wait
  done
done
wait; echo "DONE mitbih_3seed."
