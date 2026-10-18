#!/usr/bin/env bash
# DECISIVE ablation: does (i) passive Neocortex help MIT-BIH classification over no-Neo (Hippo only)?
# no_neo x3seeds e30. compare: (i) off(=passive Neo) 0.9143 ; (ii) rec ~0.81. If (i)>no_neo -> passive
# Neo earns its place on classification -> complementary 2-variant framing solid.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; DATA_ROOT="$ROOT/data"
SUMMARY_DIR="$ROOT/../results/ECG/noneo_mitbih/launcher"; mkdir -p "$SUMMARY_DIR"
source /home/anaconda3/etc/profile.d/conda.sh; conda activate snn_jelly
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<3000{print $1}')
[ "${#GPUS[@]}" -eq 0 ] && GPUS=(3 4 5); echo "free GPUs: ${GPUS[*]}"
i=0; NG=${#GPUS[@]}
for seed in 42 7 2026; do gpu=${GPUS[$(( i % NG ))]}; i=$((i+1))
  logf="$SUMMARY_DIR/mitbih_noneo_s${seed}.log"; : > "$logf"
  echo "[run] noneo seed=$seed GPU=$gpu" | tee -a "$logf"
  python3 train.py --model myModel --seed "$seed" --log_dir ../results/ECG/noneo_mitbih --gating attn --attn SSA \
    -s --test --print_epoch 1 -nd "$gpu" -e 30 --warm_up_epoch 0 --num_channels 1 -bs 512 -lr 0.01 \
    --max_lr 0.005 -emb 384 -nh 8 --mlp_ratios 1 --layers 1 --keep_ratio 0.15 --time_layers 2 \
    --alpha 0.1 --patch_size 32 --no-bias --scheduler cosine --no_neo \
    --data UEA --data_path ECG --seq_len 187 --root_path "$DATA_ROOT" >> "$logf" 2>&1 &
done
wait; echo "DONE noneo_mitbih."
