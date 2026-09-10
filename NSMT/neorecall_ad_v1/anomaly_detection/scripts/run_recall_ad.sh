#!/usr/bin/env bash
# AD REVERSAL of Neo-as-recall. In anomaly detection the Hippocampus's reconstruction error IS
# the anomaly score, so Hippo keeps reconstruction; the Neocortex is flipped to NEXT-PATCH
# PREDICTION (target = patch sequence shifted by one -- no extra data needed) and is read back
# through the additive alpha gate. Both directions stop-gradient. +1 parameter (alpha) only.
# usage: run_recall_ad.sh <DS> <cond> <seed> <gpu>
set -u
DS=${1:?}; C=${2:?}; S=${3:?}; GPU=${4:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # task root
R="$(cd "$W/../.." && pwd)"  # NSMT root
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}   # clone of snn_jelly with torch 1.12.0+cu113 restored
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
EPOCHS=${EPOCHS:-100}                                            # override to 1 for smoke tests
RES=${RES:-$W/results}; LOG=${LOG:-$W/log}  # override to keep smoke runs out of the real results
mkdir -p "$RES" "$LOG"
RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
case $C in
  baseline)    MF="" ;;
  recall)      MF="--neo_recall" ;;                                  # ★ Neo = next-patch prediction
  recall_rec)  MF="--neo_recall --no_neo_recall_next" ;;             # control: Neo back to reconstruction
  recall_none) MF="--neo_recall --neo_recall_gate none" ;;           # control: no read-back
  *) echo "bad cond $C"; exit 1 ;;
esac
case $DS in
  SMD)  CO=38; AR=0.5; LR=0.000869989; AL=0.1397 ;;
  MSL)  CO=55; AR=1.0; LR=0.0005;      AL=0.1 ;;
  PSM)  CO=25; AR=1.0; LR=0.0005;      AL=0.1 ;;
  SMAP) CO=25; AR=1.0; LR=0.0005;      AL=0.1 ;;
esac
IX="rcad_${DS}_${C}"; OUT=$LOG/${IX}_seed${S}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$W" && $PY ./train.py --model myModel --seed $S --gating attn --no-bias --scheduler reduce -s --test \
  -nd $GPU -e $EPOCHS --warm_up_epoch 0 -bs 128 -emb 32 -nh 16 --mlp_ratios 4 --keep_ratio 0 \
  --patch_size 8 --seq_len 100 --c_out $CO --data $DS --features M --pred_len 0 \
  --root_path $R/anomaly_detection/dataset/$DS \
  --anomaly_ratio $AR -lr $LR --alpha $AL --time_layers 2 --log_dir $LOG/${IX} $MF > "$OUT" 2>&1)
# point-adjusted metrics:  "adj   : acc=.. pre=.. rec=.. f1=.."
LINE=$(tail -c 6000 "$OUT" | tr -d '\000' | grep -aoE 'adj +: acc=[0-9.]+ pre=[0-9.]+ rec=[0-9.]+ f1=[0-9.]+' | tail -1)
F1=$(echo "$LINE" | grep -oE 'f1=[0-9.]+' | cut -d= -f2)
PR=$(echo "$LINE" | grep -oE 'pre=[0-9.]+' | cut -d= -f2)
RC=$(echo "$LINE" | grep -oE 'rec=[0-9.]+' | cut -d= -f2)
CK=$(find $LOG/${IX} -name 'best+model.pt' -newermt "-1 day" 2>/dev/null | head -1)
AL_=$($PY -c "
import torch
try:
    sd=torch.load('$CK',map_location='cpu'); sd=sd.get('state_dict',sd) if isinstance(sd,dict) else sd
    v=sd.get('recall_alpha'); print(f'{torch.sigmoid(v).item():.4f}' if v is not None else 'na')
except Exception: print('na')" 2>/dev/null)
t1=$(date +%s)
if [ -z "$F1" ]; then echo "SKIP(no f1) ds=$DS cond=$C seed=$S" | tee -a $RES/raw.txt; exit 1; fi
echo "RESULT exp=rcad dataset=$DS cond=$C seed=$S f1=$F1 precision=$PR recall=$RC alpha=$AL_ time_s=$((t1-t0)) sha=$SHA" | tee -a $RES/raw.txt
