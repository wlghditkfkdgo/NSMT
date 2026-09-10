#!/usr/bin/env bash
# SWaT only: data_dict key is 'SWAT' (uppercase) but dir is dataset/SWaT. noneo/passive/active x3 seeds.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source /home/anaconda3/etc/profile.d/conda.sh; conda activate snn_jelly
IFS=' ' read -r -a GPUS <<< "${SW_GPUS:-0 1 2 3}"
echo "GPUs: ${GPUS[*]}"
run_one(){ local cond="$1" seed="$2" gpu="$3"
  local ex=""; [ "$cond" = "active" ] && ex="--neo_recurrent --neo_tau 2"; [ "$cond" = "noneo" ] && ex="--no_neo"
  local ld="log/adT_${cond}"; local sd="$ROOT/$ld/launcher"; mkdir -p "$sd"
  local logf="$sd/SWaT_s${seed}.log"
  if grep -q "Test was successfully done" "$logf" 2>/dev/null; then echo "[skip] SWaT $cond s$seed"; return; fi
  : > "$logf"; echo "[run] SWaT $cond s$seed GPU=$gpu" | tee -a "$logf"
  python3 ./train.py --model myModel --gating attn --no-bias --scheduler reduce -s --test --seed "$seed" \
    -nd "$gpu" -e 30 --warm_up_epoch 0 -bs 32 -emb 32 -nh 4 -lr 0.000862736 --keep_ratio 0.25 \
    --alpha 0.3996 --time_layers 2 --mlp_ratios 2 --data SWAT --pred_len 0 --patch_size 12 $ex \
    --log_dir "$ld" --root_path "$ROOT/dataset/SWaT" --seq_len 100 --c_out 51 \
    --anomaly_ratio 1 --features M >> "$logf" 2>&1; }
JOBS=(); for cond in noneo passive active; do for seed in 42 7 2026; do JOBS+=("$cond $seed"); done; done
NG=${#GPUS[@]}
worker(){ local gpu="$1" wid="$2" k=0 job c s
  for job in "${JOBS[@]}"; do
    if (( k % NG == wid )); then read -r c s <<< "$job"; run_one "$c" "$s" "$gpu"; fi
    k=$((k+1)); done; }
for w in "${!GPUS[@]}"; do worker "${GPUS[$w]}" "$w" & done
wait; echo "DONE swat."
