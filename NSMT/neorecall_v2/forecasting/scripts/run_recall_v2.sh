#!/usr/bin/env bash
# Neo-as-recall. Hippo does self-attention + MLP and predicts; Neo reconstructs the window FROM
# Hippo's pre-prediction representation (T=N, LIF on the patch axis) and is read back through a
# gate. BOTH directions are stop-gradient, so each net is trained only by its own objective.
# alpha is sigmoid-parameterised and starts at ~0.98 => Neo begins with NO influence, so it can
# only be opted into; the learned alpha is a direct readout of whether Neo is useful.
# usage: run_recall_v2.sh <DS> <PL> <cond> <seed> <gpu>
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
AUX_NEXT=${AUX_NEXT:-}                                     # exp C: set to 1 -> aux is NEXT-patch prediction
MASK=${MASK:-}                                             # exp A: mask ratio in (0,1) -> masked reconstruction aux
AUX_L1=${AUX_L1:-}                                         # set to 1 for MAE aux (default MSE)
NEO_BACKEND=${NEO_BACKEND:-}                             # neolayer(default)|reservoir — frozen spiking liquid
RES_WREC=${RES_WREC:-}                                     # full(default)|zero — param-matched recurrence control
RES_SIZE=${RES_SIZE:-}                                     # reservoir neurons R (default 256)
PATCH=${PATCH:-8}                                          # lookback 을 늘릴 때 N 을 고정하려면 함께 키운다
SEQ_LEN=${SEQ_LEN:-96}                                      # lookback. T=N=num_patches 이므로 어텐션이 N^2 로 커진다
BS=${BS:-64}
OUT_RES=${OUT_RES:-}                                        # P3: 예측 수준 덧셈 추세 경로 (zero-init beta)
NO_AUX=${NO_AUX:-}                                          # 1 = aux 경로 자체를 제거 (--alpha 0 과 동치, 죽은 파라미터 없음)
MIXER=${MIXER:-}                                           # attn(default)|max — Hippo token mixer
SERIAL=${SERIAL:-}                                         # off(default)|fold|skip — single-path wiring
RES_READOUT=${RES_READOUT:-}                               # spike(default)|analog — readout nonlinearity
REFINE=${REFINE:-}                                         # extra Block passes under NEO_XATTN (default 1)
NEO_XATTN=${NEO_XATTN:-}                                   # 1 = Block does q=Hippo,kv=Neo cross-attn (2-pass)
ALPHA=${ALPHA:-0.5}                                        # 0 turns the auxiliary reconstruction loss OFF
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
  --gating attn --no-bias --scheduler reduce -s --test -nd $GPU --warm_up_epoch $WARMUP -bs $BS -emb 64 -nh 8 \
  --max_ratio 2 --data $DS --data_path $DS.csv --pred_len $PL --patch_size $PATCH --freq $FREQ --features M --target OT \
  --root_path $R/forecasting/dataset/ETT-small --seq_len $SEQ_LEN --patience $PATIENCE --c_in 7 \
  --alpha $ALPHA --keep_ratio 0.25 --mlp_ratios 1 -lr 0.001 --time_layers 2 -e $EPOCHS \
  --init_order_fix ${NEO_TAU:+--neo_tau $NEO_TAU} ${NEO_FULL_GRAD:+--neo_full_grad} \
  ${NEO_RECALL_GRAD:+--neo_recall_grad $NEO_RECALL_GRAD} ${AUX_NEXT:+--neo_aux_next} \
  ${MASK:+--neo_mask_ratio $MASK} ${AUX_L1:+--aux_l1} ${NEO_BACKEND:+--neo_backend $NEO_BACKEND} \
  ${RES_WREC:+--res_wrec $RES_WREC} ${RES_SIZE:+--res_size $RES_SIZE} ${NEO_XATTN:+--neo_xattn} ${REFINE:+--refine_steps $REFINE} ${RES_READOUT:+--res_readout $RES_READOUT} ${MIXER:+--hippo_mixer $MIXER} ${SERIAL:+--serial_mode $SERIAL} ${NO_AUX:+--no_aux} ${OUT_RES:+--neo_out_residual} $MF > "$OUT" 2>&1)
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
    v=sd.get('recall_alpha')
    if v is not None: print(f'{torch.sigmoid(v).item():.4f}')
    else:
        g=sd.get('serial_gamma')          # serial 'skip': zero-init additive gate on the reservoir
        print(f'g{torch.tanh(g).item():+.4f}' if g is not None else 'na')
except Exception: print('na')" 2>/dev/null)
if [ -z "$MSE" ]; then echo "SKIP(no mse) ds=$DS pl=$PL cond=${C}${TAG:+_$TAG} seed=$S" | tee -a $RES/raw.txt; exit 1; fi
echo "RESULT exp=recall dataset=${DS}_p${PL} cond=${C}${TAG:+_$TAG} seed=$S mse=$MSE alpha=$AL params=$PAR time_s=$((t1-t0)) sha=$SHA" | tee -a $RES/raw.txt
