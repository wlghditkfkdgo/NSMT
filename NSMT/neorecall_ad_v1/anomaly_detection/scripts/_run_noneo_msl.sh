#!/usr/bin/env bash
# MSL no-Neo ablation (Hippo only) for the complementary table. e30, 3 seeds.
# compare: (i) off(passive) 0.7783 ; (ii) rec 0.7826-0.7854. expect (ii)>(i)~=no-Neo.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; DATA_ROOT="$ROOT/dataset/MSL"
SUMMARY_DIR="$ROOT/log/noneo_msl/launcher"; mkdir -p "$SUMMARY_DIR"
source /home/anaconda3/etc/profile.d/conda.sh; conda activate snn_jelly
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<3000{print $1}')
[ "${#GPUS[@]}" -eq 0 ] && GPUS=(1 2 3 4)
echo "free GPUs: ${GPUS[*]}"
i=0; NG=${#GPUS[@]}
for seed in 42 7 2026; do
  gpu=${GPUS[$(( i % NG ))]}; i=$((i+1))
  logf="$SUMMARY_DIR/MSL_noneo_s${seed}.log"; : > "$logf"
  echo "[run] noneo MSL seed=$seed GPU=$gpu" | tee -a "$logf"
  python3 ./train.py --model myModel --gating attn --no-bias --scheduler reduce -s --test --seed "$seed" \
    -nd "$gpu" -e 30 --warm_up_epoch 0 -bs 64 -emb 32 -nh 16 -lr 5.5842e-05 --keep_ratio 0.25 --alpha 0.3279 \
    --time_layers 1 --mlp_ratios 2 --data MSL --pred_len 0 --patch_size 12 --no_neo \
    --root_path "$DATA_ROOT" --seq_len 100 --c_out 55 --anomaly_ratio 1 --features M >> "$logf" 2>&1 &
done
wait; echo "DONE noneo_msl."
