#!/usr/bin/env bash
# Validate recurrent Neo (P1) on MIT-BIH ECG classification. {off, rec_nt2, rec_nt8} x3seeds. macro-F1.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; DATA_ROOT="$ROOT/data"
SUMMARY_DIR="$ROOT/../results/ECG/rec_mitbih/launcher"; mkdir -p "$SUMMARY_DIR"
source /home/anaconda3/etc/profile.d/conda.sh; conda activate snn_jelly
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1}')
[ "${#GPUS[@]}" -eq 0 ] && GPUS=(0 1 2 3 4 5); echo "free GPUs: ${GPUS[*]}"
run_one(){ local seed="$1" gpu="$2" mode="$3"
  local ex=""; case "$mode" in rec_nt2) ex="--neo_recurrent --neo_tau 2";; rec_nt8) ex="--neo_recurrent --neo_tau 8";; off) ex="--neo_tau 2";; esac
  local logf="$SUMMARY_DIR/mitbih_${mode}_s${seed}.log"; : > "$logf"
  echo "[run] mitbih $mode seed=$seed GPU=$gpu" | tee -a "$logf"
  python3 train.py --model myModel --seed "$seed" --log_dir ../results/ECG/rec_mitbih --gating attn --attn SSA \
    -s --test --print_epoch 1 -nd "$gpu" -e 30 --warm_up_epoch 0 --num_channels 1 -bs 512 -lr 0.01 \
    --max_lr 0.005 -emb 384 -nh 8 --mlp_ratios 1 --layers 1 --keep_ratio 0.15 --time_layers 2 \
    --alpha 0.1 --patch_size 32 --no-bias --scheduler cosine $ex \
    --data UEA --data_path ECG --seq_len 187 --root_path "$DATA_ROOT" >> "$logf" 2>&1; }
i=0; NG=${#GPUS[@]}
for mode in off rec_nt2 rec_nt8; do for seed in 42 7 2026; do gpu=${GPUS[$(( i % NG ))]}; run_one "$seed" "$gpu" "$mode" & i=$((i+1)); (( i % NG == 0 )) && wait; done; done
wait; echo "DONE rec_mitbih."
