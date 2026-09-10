#!/usr/bin/env bash
# Neo-as-recall. Hippo does self-attention + MLP and predicts; Neo reconstructs the window FROM
# Hippo's pre-prediction representation (T=N, LIF on the patch axis) and is read back through a
# gate. BOTH directions are stop-gradient, so each net is trained only by its own objective.
# alpha is sigmoid-parameterised and starts at ~0.98 => Neo begins with NO influence, so it can
# only be opted into; the learned alpha is a direct readout of whether Neo is useful.
# usage: run_recall.sh <DS> <PL> <cond> <seed> <gpu>
set -u
DS=${1:?}; PL=${2:?}; C=${3:?}; S=${4:?}; GPU=${5:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # task root
R="$(cd "$W/../.." && pwd)"  # NSMT root
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}   # clone of snn_jelly with torch 1.12.0+cu113 restored
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EPOCHS=${EPOCHS:-50}                                       # override to 1 for smoke tests
PATIENCE=${PATIENCE:-3}; WARMUP=${WARMUP:-0}               # training-schedule sweep (structure unchanged)
NEO_TAU=${NEO_TAU:-}                                       # empty = config default (8.0); set to pass --neo_tau
NEO_FULL_GRAD=${NEO_FULL_GRAD:-}                           # set to 1 to remove the 0.05 grad throttle (ours.py:269)
NEO_RECALL_GRAD=${NEO_RECALL_GRAD:-}                       # stop(default)|full|to_neo|to_hippo — Hippo<->Neo interface
TAG=${TAG:-}                                               # suffixes cond + log dir so sweeps stay distinguishable
RES=${RES:-$W/results}; LOG=${LOG:-$W/log}  # override to keep smoke runs out of the real results
mkdir -p "$RES" "$LOG"
RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
case $C in
  baseline)     MF="" ;;
  recall_raw)   MF="--neo_recall --neo_recall_tgt raw" ;;
  recall_low)   MF="--neo_recall --neo_recall_tgt lowfreq" ;;
  recall_mul)   MF="--neo_recall --neo_recall_gate mul" ;;
  recall_none)  MF="--neo_recall --neo_recall_gate none" ;;   # control: self-attn change only, no gate
  *) echo "bad cond $C"; exit 1 ;;
esac
case $DS in ETTh1|ETTh2) FREQ=h ;; ETTm1|ETTm2) FREQ=t ;; esac
IX="rc_${DS}_p${PL}_${C}${TAG:+_$TAG}"; OUT=$LOG/${IX}_seed${S}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$W" && $PY ./train.py --log_dir $LOG/${IX} --model myModel --seed $S \
  --gating attn --no-bias --scheduler reduce -s --test -nd $GPU --warm_up_epoch $WARMUP -bs 64 -emb 64 -nh 8 \
  --max_ratio 2 --data $DS --data_path $DS.csv --pred_len $PL --patch_size 8 --freq $FREQ --features M --target OT \
  --root_path $R/forecasting/dataset/ETT-small --seq_len 96 --patience $PATIENCE --c_in 7 \
  --alpha 0.5 --keep_ratio 0.25 --mlp_ratios 1 -lr 0.001 --time_layers 2 -e $EPOCHS \
  --init_order_fix ${NEO_TAU:+--neo_tau $NEO_TAU} ${NEO_FULL_GRAD:+--neo_full_grad} \
  ${NEO_RECALL_GRAD:+--neo_recall_grad $NEO_RECALL_GRAD} $MF > "$OUT" 2>&1)
MSE=$(grep -aE 'mse_overall' "$OUT"|tail -1|grep -oE '[0-9.]+')
PAR=$(grep -a 'number of parameters' "$OUT"|grep -oE '[0-9]+'|tail -1); t1=$(date +%s)
# learned alpha = sigmoid(recall_alpha) from the best checkpoint
# seeds share $LOG/$IX (config.py builds a `seed<N>_...` subdir per run), so the path MUST be
# filtered by seed -- `find | head -1` picked an arbitrary seed's checkpoint and mis-attributed alpha.
CK=$(find $LOG/${IX} -path "*seed${S}_*" -name 'best+model.pt' -newermt "-1 day" 2>/dev/null | head -1)
AL=$($PY -c "
import torch,sys
try:
    sd=torch.load('$CK',map_location='cpu'); sd=sd.get('state_dict',sd) if isinstance(sd,dict) else sd
    v=sd.get('recall_alpha');  print(f'{torch.sigmoid(v).item():.4f}' if v is not None else 'na')
except Exception: print('na')" 2>/dev/null)
if [ -z "$MSE" ]; then echo "SKIP(no mse) ds=$DS pl=$PL cond=$C seed=$S" | tee -a $RES/raw.txt; exit 1; fi
echo "RESULT exp=recall dataset=${DS}_p${PL} cond=${C}${TAG:+_$TAG} seed=$S mse=$MSE alpha=$AL params=$PAR time_s=$((t1-t0)) sha=$SHA" | tee -a $RES/raw.txt
