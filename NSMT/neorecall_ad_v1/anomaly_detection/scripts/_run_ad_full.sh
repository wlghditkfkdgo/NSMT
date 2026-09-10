#!/usr/bin/env bash
# Complete the AD table: {no_neo, passive(off), active(rec tau2)} x {PSM,SMAP,SMD,SWaT} x 3 seeds.
# distinct log_dir per condition (AD CSV path is hyperparam-based -> same-dataset conditions collide).
# point-adj F1 (adj_f-score = CSV col 7). MAX 4 GPUs. (MSL already done elsewhere.)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
source /home/anaconda3/etc/profile.d/conda.sh; conda activate snn_jelly
# data c_out bs emb nh lr alpha keep tl mlp patch anomaly_ratio
declare -A CFG
CFG[PSM]="25 128 32 4 0.000862736 0.3996 0.25 4 6 8 1"
CFG[SMAP]="25 32 32 4 0.000423568 0.1957 0.10 1 2 12 1"
CFG[SMD]="38 128 32 16 0.000869989 0.1397 0 2 4 8 0.5"
CFG[SWaT]="51 32 32 4 0.000862736 0.3996 0.25 2 2 12 1"
IFS=' ' read -r -a GPUS <<< "${AD_GPUS:-0}"   # GPUs from env AD_GPUS (default "0")
echo "using GPUs: ${GPUS[*]}"
run_one(){ local data="$1" cond="$2" seed="$3" gpu="$4"
  IFS=' ' read -r cout bs emb nh lr al kr tl mr ps ar <<< "${CFG[$data]}"
  local ex=""; [ "$cond" = "active" ] && ex="--neo_recurrent --neo_tau 2"; [ "$cond" = "noneo" ] && ex="--no_neo"
  local ld="log/adT_${cond}"; local sd="$ROOT/$ld/launcher"; mkdir -p "$sd"
  local logf="$sd/${data}_s${seed}.log"
  if grep -q "Test was successfully done" "$logf" 2>/dev/null; then echo "[skip-done] $data $cond s$seed"; return; fi
  : > "$logf"
  echo "[run] $data $cond seed=$seed GPU=$gpu" | tee -a "$logf"
  python3 ./train.py --model myModel --gating attn --no-bias --scheduler reduce -s --test --seed "$seed" \
    -nd "$gpu" -e 30 --warm_up_epoch 0 -bs "$bs" -emb "$emb" -nh "$nh" -lr "$lr" --keep_ratio "$kr" \
    --alpha "$al" --time_layers "$tl" --mlp_ratios "$mr" --data "$data" --pred_len 0 --patch_size "$ps" $ex \
    --log_dir "$ld" --root_path "$ROOT/dataset/$data" --seq_len 100 --c_out "$cout" \
    --anomaly_ratio "$ar" --features M >> "$logf" 2>&1; }
JOBS=(); for data in PSM SMAP SMD SWaT; do for cond in noneo passive active; do for seed in 42 7 2026; do JOBS+=("$data $cond $seed"); done; done; done
NG=${#GPUS[@]}
# one worker per GPU: each handles every NG-th job sequentially on its GPU (no barrier -> full util)
worker(){ local gpu="$1" wid="$2" k=0 job d c s
  for job in "${JOBS[@]}"; do
    if (( k % NG == wid )); then read -r d c s <<< "$job"; run_one "$d" "$c" "$s" "$gpu"; fi
    k=$((k+1))
  done
}
for w in "${!GPUS[@]}"; do worker "${GPUS[$w]}" "$w" & done
wait; echo "DONE ad_full."
