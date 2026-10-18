#!/usr/bin/env bash
# Fix recurrent Neo classification instability (MIT-BIH). P1=recurrence gate (g init 0, stable superset),
# P2=gradient clipping. conditions: off | rec(no fix) | rec+gate | rec+clip | rec+gate+clip. neo_tau8, e30, 3seeds.
# baseline off ~0.914, broken rec ~0.57-0.75. target: gate/clip recover to >= off.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; DATA_ROOT="$ROOT/data"
SUMMARY_DIR="$ROOT/../results/ECG/rec_stab/launcher"; mkdir -p "$SUMMARY_DIR"
source /home/anaconda3/etc/profile.d/conda.sh; conda activate snn_jelly
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<3000{print $1}')
[ "${#GPUS[@]}" -eq 0 ] && GPUS=(3 4 5); echo "free GPUs: ${GPUS[*]}"
run_one(){ local seed="$1" gpu="$2" mode="$3"
  local ex=""
  case "$mode" in
    off) ex="--neo_tau 2";;
    rec) ex="--neo_recurrent --neo_tau 8";;
    gate) ex="--neo_recurrent --neo_tau 8 --neo_rec_gate";;
    clip) ex="--neo_recurrent --neo_tau 8 --grad_clip 1.0";;
    gateclip) ex="--neo_recurrent --neo_tau 8 --neo_rec_gate --grad_clip 1.0";;
  esac
  local logf="$SUMMARY_DIR/mitbih_${mode}_s${seed}.log"; : > "$logf"
  echo "[run] stab $mode seed=$seed GPU=$gpu" | tee -a "$logf"
  python3 train.py --model myModel --seed "$seed" --log_dir ../results/ECG/rec_stab --gating attn --attn SSA \
    -s --test --print_epoch 1 -nd "$gpu" -e 30 --warm_up_epoch 0 --num_channels 1 -bs 512 -lr 0.01 \
    --max_lr 0.005 -emb 384 -nh 8 --mlp_ratios 1 --layers 1 --keep_ratio 0.15 --time_layers 2 \
    --alpha 0.1 --patch_size 32 --no-bias --scheduler cosine $ex \
    --data UEA --data_path ECG --seq_len 187 --root_path "$DATA_ROOT" >> "$logf" 2>&1; }
i=0; NG=${#GPUS[@]}
for mode in off rec gate clip gateclip; do for seed in 42 7 2026; do gpu=${GPUS[$(( i % NG ))]}; run_one "$seed" "$gpu" "$mode" & i=$((i+1)); (( i % NG == 0 )) && wait; done; done
wait; echo "DONE rec_stab."
