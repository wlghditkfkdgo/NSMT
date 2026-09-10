#!/usr/bin/env bash
# Validate recurrent Neo (P1, long-term memory) on MSL anomaly detection. {off, rec_nt2, rec_nt8} x3seeds.
# point-adjusted F1 (higher better) + energy auto-logged. Does the Neo accuracy win transfer to AD?
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; DATA_ROOT="$ROOT/dataset/MSL"
SUMMARY_DIR="$ROOT/log/rec_msl/launcher"; mkdir -p "$SUMMARY_DIR"
source /home/anaconda3/etc/profile.d/conda.sh; conda activate snn_jelly
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<2000{print $1}')
[ "${#GPUS[@]}" -eq 0 ] && GPUS=(0 1 2 3 4 5); echo "free GPUs: ${GPUS[*]}"
run_one(){ local seed="$1" gpu="$2" mode="$3"
  local ex=""; case "$mode" in rec_nt2) ex="--neo_recurrent --neo_tau 2";; rec_nt8) ex="--neo_recurrent --neo_tau 8";; off) ex="--neo_tau 2";; esac
  local logf="$SUMMARY_DIR/MSL_${mode}_s${seed}.log"; : > "$logf"
  echo "[run] MSL $mode seed=$seed GPU=$gpu" | tee -a "$logf"
  python3 ./train.py --model myModel --gating attn --no-bias --scheduler reduce -s --test --seed "$seed" \
    -nd "$gpu" -e 100 --warm_up_epoch 0 -bs 64 -emb 32 -nh 16 -lr 5.5842e-05 --keep_ratio 0.25 --alpha 0.3279 \
    --time_layers 1 --mlp_ratios 2 --data MSL --pred_len 0 --patch_size 12 $ex \
    --root_path "$DATA_ROOT" --seq_len 100 --c_out 55 --anomaly_ratio 1 --features M >> "$logf" 2>&1; }
i=0; NG=${#GPUS[@]}
for mode in off rec_nt2 rec_nt8; do for seed in 42 7 2026; do gpu=${GPUS[$(( i % NG ))]}; run_one "$seed" "$gpu" "$mode" & i=$((i+1)); (( i % NG == 0 )) && wait; done; done
wait; echo "DONE rec_msl."
