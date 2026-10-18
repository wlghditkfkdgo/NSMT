#!/usr/bin/env bash
# Clean MIT-BIH re-run {no-Neo,passive,active} x 3 seeds with per-class F1 + confusion (R5).
# distinct log_dir per condition. GPU 3 (single).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"; DATA_ROOT="$ROOT/data"
source /home/anaconda3/etc/profile.d/conda.sh; conda activate snn_jelly
run_one(){ local cond="$1" seed="$2"
  local ex=""; case "$cond" in
    noneo) ex="--no_neo";; passive) ex="--neo_tau 2";; active) ex="--neo_recurrent --neo_tau 8";; esac
  local ld="../results/ECG/ecgF_${cond}"; local sd="$ROOT/$ld/launcher"; mkdir -p "$sd"; local logf="$sd/s${seed}.log"
  grep -q "PERCLASS_F1" "$logf" 2>/dev/null && { echo "[skip] $cond s$seed"; return; }
  : > "$logf"; echo "[run] $cond s$seed" | tee -a "$logf"
  python3 train.py --model myModel --seed "$seed" --log_dir "$ld" --gating attn --attn SSA \
    -s --test --print_epoch 30 -nd 3 -e 30 --warm_up_epoch 0 --num_channels 1 -bs 512 -lr 0.01 \
    --max_lr 0.005 -emb 384 -nh 8 --mlp_ratios 1 --layers 1 --keep_ratio 0.15 --time_layers 2 \
    --alpha 0.1 --patch_size 32 --no-bias --scheduler cosine $ex \
    --data UEA --data_path ECG --seq_len 187 --root_path "$DATA_ROOT" >> "$logf" 2>&1; }
for cond in noneo passive active; do for seed in 42 7 2026; do run_one "$cond" "$seed"; done; done
echo "DONE ecg_perclass."
