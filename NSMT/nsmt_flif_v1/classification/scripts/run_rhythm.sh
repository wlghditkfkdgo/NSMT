#!/usr/bin/env bash
# Stage 9 (MODI_PROJECT_LOG 2026-10-19): one run on CinC 2017 or SHD with the protocols adopted in stage 8.
# usage: DATA=CINC2017|SHD COND=<cond> [FOLD=0] [LR=...] [EPOCHS=...] [CROP=k] run_rhythm.sh <seed> <gpu>
#   COND: nsmt_lif | nsmt_flif | nsmt_pop | spk_ssa | mix_flif_bidir | mix_pop_bidir | mix_lif_bidir | mix_none
#         (+ suffix _ce for plain CE on CinC 2017, e.g. nsmt_lif_ce, mix_lif_bidir_ce)
#   common: D 384, 8 heads, mlp 1, 1 block, 2 neocortex layers, keep_ratio 0.15, aux 0.1, tau 2, AdamW wd 1e-4,
#           timm cosine, batch 32; CinC: patch 64 (117 tokens), 30 epochs; SHD: patch 2 (100 tokens), 50 epochs.
set -u
S=${1:?}; GPU=${2:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
DATA=${DATA:?DATA=CINC2017|SHD}; COND=${COND:?}; FOLD=${FOLD:-0}; CROP=${CROP:-0}
BASE=${COND%_ce}; LOSS=cal; [ "$BASE" != "$COND" ] && LOSS=ce
case $BASE in
  nsmt_lif)       CF="--model myModel --neo_neuron lif --neo_where both" ;;
  nsmt_flif)      CF="--model myModel --neo_neuron flif --neo_where both --neo_alpha 0.5" ;;
  nsmt_pop)       CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0" ;;
  spk_ssa)        CF="--model myModel --hippo_ssa --hippo_T 4" ;;
  mix_flif_bidir) CF="--model neoMixer --mixer flif --mixer_bidir --neo_alpha 0.5" ;;
  mix_pop_bidir)  CF="--model neoMixer --mixer pop_flif --mixer_bidir --neo_alpha 0.5 --pop_soma_init 1.0" ;;
  mix_lif_bidir)  CF="--model neoMixer --mixer lif --mixer_bidir" ;;
  mix_none)       CF="--model neoMixer --mixer none" ;;
  *) echo "bad cond $COND"; exit 1 ;;
esac
case $DATA in
  CINC2017) HP="--patch_size 64 -e ${EPOCHS:-30} --loss_kind $LOSS --fold $FOLD" ;;
  SHD)      HP="--patch_size 2 -e ${EPOCHS:-50}"; [ "$LOSS" = ce ] && { echo "_ce is CinC-only"; exit 1; } ;;
  *) echo "bad data $DATA"; exit 1 ;;
esac
[ "$CROP" != 0 ] && HP="$HP --crop_tokens $CROP"
LR=${LR:-0.001}
SUITE=${SUITE:-flif-v1-rhythm-$(date +%Y%m%d)}
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}
mkdir -p "$RES" "$LOG"; RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
TAGC=${COND}$([ "$CROP" != 0 ] && echo "_crop$CROP")
OUT=$LOG/${DATA}_${TAGC}_f${FOLD}_seed${S}_lr${LR}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train_rhythm.py \
  --data $DATA --seed $S --log_dir "$LOG/$DATA" --gating attn -s -nd 0 --warm_up_epoch 0 \
  -emb 384 -nh 8 --mlp_ratios 1 --layers 1 --time_layers 2 --keep_ratio 0.15 --alpha 0.1 --tau 2.0 --no-bias \
  --scheduler cosine -bs 32 --eval_bs 32 -lr $LR --weight_decay 1e-4 --max_lr 0.005 $HP $CF --tag $TAGC lr$LR seed$S > "$OUT" 2>&1)
RC=$?
D=$(ls -td "$LOG/$DATA"/*/*"+${TAGC}+lr${LR}+seed${S}+"*/seed${S}_* 2>/dev/null | head -1)
CSV="$D/log/final+result.csv"; LCSV="$D/log/last/final+result.csv"
if [ $RC -ne 0 ] || [ ! -s "$CSV" ]; then echo "SKIP(rc=$RC, no result) data=$DATA cond=$COND fold=$FOLD seed=$S lr=$LR crop=$CROP out=$OUT" | tee -a "$RES/raw.txt"; exit 1; fi
VALS=$($PY - "$CSV" "$LCSV" <<'EOF'
import csv, sys
b = next(csv.DictReader(open(sys.argv[1]))); l = next(csv.DictReader(open(sys.argv[2])))
keys = [k for k in b if k not in ('weights',)]
print(' '.join(f'{k}={b[k]}' for k in keys) + ' ' + ' '.join(f'last_{k}={l[k]}' for k in ('score', 'acc', 'macro_f1')))
EOF
)
BE=$(cat "$D/log/best_epoch.txt" 2>/dev/null | tr ' ' ':')
PAR=$(grep -a 'number of parameters' "$OUT" | grep -oE 'parameters : [0-9]+' | grep -oE '[0-9]+'); t1=$(date +%s)
echo "RESULT exp=nsmt_flif_v1_rhythm data=$DATA cond=$COND fold=$FOLD seed=$S lr=$LR crop=$CROP $VALS best_epoch_val=$BE params=$PAR time_s=$((t1-t0)) sha=$SHA dir=$D" | tee -a "$RES/raw.txt"
