#!/usr/bin/env bash
# NSMT paper baseline (nsmt_base): one ETT forecasting run with the settings that reproduce the
# paper's Table 13 (see NSMT/papers/NSMT_code_mapping.md §3; read from the saved paper runs, not
# from the sweep-state upstream scripts). Code = origin/main:forecasting (published repository).
# usage: run_ett.sh <DS: ETTh1|ETTh2|ETTm1|ETTm2> <PL> <seed> <gpu>
set -u
DS=${1:?}; PL=${2:?}; S=${3:?}; GPU=${4:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # task root (nsmt_base/forecasting)
R="$(cd "$W/../.." && pwd)"  # NSMT root
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}   # Python 3.10, torch 1.12.0+cu113, spikingjelly cupy backend
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SUITE=${SUITE:-ett-paper-check-20261014}
EPOCHS=${EPOCHS:-}                                         # empty = the paper value per dataset
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}
mkdir -p "$RES" "$LOG"
RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
case $DS in                       # patch k emb heads alpha mlp M lr bs epoch patience
  ETTh1) HP="--patch_size 8 --keep_ratio 0.25 -emb 32 -nh 2  --alpha 0.8 --mlp_ratios 1   --time_layers 2 -lr 0.0005 -bs 64 --patience 3"; E=50 ;;
  ETTh2) HP="--patch_size 8 --keep_ratio 0.3  -emb 64 -nh 8  --alpha 0.8 --mlp_ratios 0.5 --time_layers 1 -lr 0.0005 -bs 64 --patience 2"; E=100 ;;
  ETTm1) HP="--patch_size 8 --keep_ratio 0.25 -emb 64 -nh 8  --alpha 0.5 --mlp_ratios 1   --time_layers 2 -lr 0.001  -bs 64 --patience 3"; E=50 ;;
  ETTm2) HP="--patch_size 8 --keep_ratio 0.25 -emb 64 -nh 16 --alpha 0.5 --mlp_ratios 1   --time_layers 2 -lr 0.0005 -bs 64 --patience 2"; E=50 ;;
  *) echo "bad dataset $DS"; exit 1 ;;
esac
E=${EPOCHS:-$E}
IX="ett_${DS}_p${PL}"; OUT=$LOG/${IX}_seed${S}_${TS}_${SHA}.stdout; t0=$(date +%s)
# --log_dir is resolved relative to the code folder by config.py; keep runs of this suite together.
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train.py \
  --log_dir "$LOG" --model myModel --seed $S --gating attn --no-bias --scheduler reduce -s --test -nd 0 \
  -e $E --warm_up_epoch 0 --layers 1 --max_ratio 2 --tau 2.0 \
  --data $DS --data_path $DS.csv --root_path "$W/dataset/ETT-small" --seq_len 96 --pred_len $PL --c_in 7 \
  --features M --target OT $HP > "$OUT" 2>&1)
RC=$?
# final+result.csv of this run: mse,mae of the test split (first data row)
D=$(ls -td "$LOG/$DS"/*/*"+pred_len+${PL}+"* 2>/dev/null | head -1)
CSV="$D/log/final+result.csv"
MSE=$(sed -n 2p "$CSV" 2>/dev/null | cut -d, -f3); MAE=$(sed -n 2p "$CSV" 2>/dev/null | cut -d, -f4)
PAR=$(grep -a 'number of parameters' "$OUT" | grep -oE '[0-9]+' | tail -1); EP=$(grep -ac '^epoch' "$OUT"); t1=$(date +%s)
if [ $RC -ne 0 ] || [ -z "$MSE" ]; then echo "SKIP(rc=$RC, no result) ds=$DS pl=$PL seed=$S" | tee -a "$RES/raw.txt"; exit 1; fi
echo "RESULT exp=nsmt_base dataset=${DS}_p${PL} seed=$S mse=$MSE mae=$MAE epochs=$E params=$PAR time_s=$((t1-t0)) sha=$SHA dir=$D" | tee -a "$RES/raw.txt"
