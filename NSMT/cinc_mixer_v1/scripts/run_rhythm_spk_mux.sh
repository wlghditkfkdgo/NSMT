#!/usr/bin/env bash
# run_rhythm_spk_mux.sh (2026-10-30, branch exp/f-lif-pop-v3): copy of run_rhythm_spk.sh (left untouched: the B2 / C pool runs it)
# that also accepts the fusion MUX combs muxor / muxlin (spk_mixer.py, 10-30). Training arguments unchanged.
# run_rhythm_spk.sh (2026-10-29, branch exp/f-lif-pop-v3, NSMT/cinc_mixer_v1): one CinC 2017 run of the SpkMixer
# (Spikformer-style model, SSA -> bidirectional token-axis Neo-mixer, the chosen neuron in EVERY layer; spk_mixer.py).
# Same training / evaluation arguments as run_rhythm_sd.sh (the whole Neo-mixer series): D 384, 8 heads, mlp 1, 1 block,
# 2 mixer layers, tau 2, AdamW wd 1e-4, timm cosine, batch 32, patch 64 (117 tokens), 30 epochs, CAL-weighted CE, T_h 4.
# usage: COND=spk_<neuron>_<comb>_<cf> [FOLD=0] [LR=...] [EPOCHS=...] [SUITE=...] run_rhythm_spk.sh <seed> <gpu>
#   neuron: lif | flif | pop_lif (soma init 0.5) | pop_flif (soma init 1.0) | popg_lif | popg_flif
#   comb:   or | sdor | sdlin | linmx | muxor | muxlin            cf: iand | or | add
set -u
S=${1:?}; GPU=${2:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
DATA=CINC2017; COND=${COND:?}; FOLD=${FOLD:-0}
case $COND in spk_*) ;; *) echo "bad cond $COND"; exit 1 ;; esac
CF=${COND##*_}; REST=${COND%_*}; COMB=${REST##*_}; NK=${REST%_*}; NK=${NK#spk_}
case $NK in
  lif)       NF="--mixer lif" ;;
  flif)      NF="--mixer flif --neo_alpha 0.5" ;;
  pop_lif)   NF="--mixer pop_lif --pop_soma_init 0.5" ;;
  pop_flif)  NF="--mixer pop_flif --neo_alpha 0.5 --pop_soma_init 1.0" ;;
  popg_lif)  NF="--mixer popg_lif" ;;
  popg_flif) NF="--mixer popg_flif --neo_alpha 0.5" ;;
  *) echo "bad neuron in $COND"; exit 1 ;;
esac
case $COMB in or|sdor|sdlin|linmx|muxor|muxlin) ;; *) echo "bad comb in $COND"; exit 1 ;; esac
case $CF in iand) CFA=IAND ;; or) CFA=OR ;; add) CFA=ADD ;; *) echo "bad connect in $COND"; exit 1 ;; esac
CF_ARGS="--model spkMixer $NF --mixer_comb $COMB --connect_f $CFA --hippo_T 4"
HP="--patch_size 64 -e ${EPOCHS:-30} --loss_kind cal --fold $FOLD"
LR=${LR:-0.001}
SUITE=${SUITE:-flif-v1-rhythm-spk-$(date +%Y%m%d)}
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}
mkdir -p "$RES" "$LOG"; RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
TAGC=$COND
OUT=$LOG/${DATA}_${TAGC}_f${FOLD}_seed${S}_lr${LR}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train_rhythm.py \
  --data $DATA --seed $S --log_dir "$LOG/$DATA" --gating attn -s -nd 0 --warm_up_epoch 0 \
  -emb 384 -nh 8 --mlp_ratios 1 --layers 1 --time_layers 2 --keep_ratio 0.15 --alpha 0.1 --tau 2.0 --no-bias \
  --scheduler cosine -bs ${BS:-32} --eval_bs ${BS:-32} -lr $LR --weight_decay 1e-4 --max_lr 0.005 $HP $CF_ARGS --tag $TAGC lr$LR seed$S > "$OUT" 2>&1)
RC=$?
D=$(ls -td "$LOG/$DATA"/*/*"+${TAGC}+lr${LR}+seed${S}+"*/seed${S}_*_f${FOLD} 2>/dev/null | head -1)
CSV="$D/log/final+result.csv"; LCSV="$D/log/last/final+result.csv"
if [ $RC -ne 0 ] || [ ! -s "$CSV" ]; then echo "SKIP(rc=$RC, no result) data=$DATA cond=$COND fold=$FOLD seed=$S lr=$LR crop=0 out=$OUT" | tee -a "$RES/raw.txt"; exit 1; fi
VALS=$($PY - "$CSV" "$LCSV" <<'EOF'
import csv, sys
b = next(csv.DictReader(open(sys.argv[1]))); l = next(csv.DictReader(open(sys.argv[2])))
keys = [k for k in b if k not in ('weights',)]
print(' '.join(f'{k}={b[k]}' for k in keys) + ' ' + ' '.join(f'last_{k}={l[k]}' for k in ('score', 'acc', 'macro_f1')))
EOF
)
BE=$(cat "$D/log/best_epoch.txt" 2>/dev/null | tr ' ' ':')
PAR=$(grep -a 'number of parameters' "$OUT" | grep -oE 'parameters : [0-9]+' | grep -oE '[0-9]+'); t1=$(date +%s)
echo "RESULT exp=nsmt_flif_v1_rhythm data=$DATA cond=$COND fold=$FOLD seed=$S lr=$LR crop=0 $VALS best_epoch_val=$BE params=$PAR time_s=$((t1-t0)) sha=$SHA dir=$D" | tee -a "$RES/raw.txt"
