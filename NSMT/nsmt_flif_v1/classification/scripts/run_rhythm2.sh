#!/usr/bin/env bash
# Stage 9 (MODI_PROJECT_LOG 2026-10-19): one run on CinC 2017 or SHD with the protocols adopted in stage 8.
# run_rhythm2.sh (MODI 2026-10-21): copy of run_rhythm.sh (left untouched while pools execute it) + the BiX / soma-less
# population / population LIF / attention-BN conditions and the cause-separation population completions.
# usage: DATA=CINC2017|SHD COND=<cond> [FOLD=0] [LR=...] [EPOCHS=...] [CROP=k] run_rhythm.sh <seed> <gpu>
#   COND: nsmt_lif | nsmt_flif | nsmt_pop | spk_ssa | mix_flif_bidir | mix_pop_bidir | mix_lif_bidir | mix_none
#         (+ suffix _ce for plain CE on CinC 2017, e.g. nsmt_lif_ce, mix_lif_bidir_ce)
#         NSMT aux-loss options (MODI 2026-10-20), suffix on nsmt_*: _auxoff (alpha 0 = plain CE), _full (keep_ratio 0 =
#         reconstruct the full patched signal, no DCT low-pass), _a05 / _a09 (alpha 0.5 / 0.9); default alpha 0.1, keep 0.15
#         cause-separation conds (MODI 2026-10-21): spk_ssa_nope (no relative bias), nsmt_lif_pe / mix_lif_bidir_pe (learned
#         absolute position), mix_lif_rec_bidir (recurrent spikingjelly LIF mixer) + its control mix_sjlif_bidir, mix_pop_bidir_pt16
#         (branch taus 16/32/64/128, soma init PT16_PSI_<DATA>), nsmt_lif_t4 / nsmt_pop_t4 (hippocampus T=4 + token cross-attention);
#         suffix _psi<v> on nsmt_pop / mix_pop_bidir overrides the soma init (e.g. nsmt_pop_psi1.5)
#   common: D 384, 8 heads, mlp 1, 1 block, 2 neocortex layers, keep_ratio 0.15, aux 0.1 (unless a suffix above), tau 2, AdamW wd 1e-4,
#           timm cosine, batch 32; CinC: patch 64 (117 tokens), 30 epochs; SHD: patch 2 (100 tokens), 50 epochs.
set -u
S=${1:?}; GPU=${2:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
DATA=${DATA:?DATA=CINC2017|SHD}; COND=${COND:?}; FOLD=${FOLD:-0}; CROP=${CROP:-0}
ALPHA=0.1; KEEP=0.15; PSI=1.0
case $COND in *_psi*) PSI=${COND##*_psi}; COND=${COND%_psi*}; PSITAG=_psi$PSI ;; esac
PT16_PSI_CINC2017=${PT16_PSI_CINC2017:-2.0}; PT16_PSI_SHD=${PT16_PSI_SHD:-2.0}    # check c24 rule ({1,2,4} closest to the tau 4-32 / psi 1 initial firing): 2.0 for both (MODI 2026-10-21)
case $COND in
  nsmt_*_auxoff) ALPHA=0;   COND0=${COND%_auxoff} ;;
  nsmt_*_full)   KEEP=0;    COND0=${COND%_full} ;;
  nsmt_*_a05)    ALPHA=0.5; COND0=${COND%_a05} ;;
  nsmt_*_a09)    ALPHA=0.9; COND0=${COND%_a09} ;;
  *)             COND0=$COND ;;
esac
BASE=${COND0%_ce}; LOSS=cal; [ "$BASE" != "$COND0" ] && LOSS=ce
case $BASE in
  nsmt_lif)       CF="--model myModel --neo_neuron lif --neo_where both" ;;
  nsmt_flif)      CF="--model myModel --neo_neuron flif --neo_where both --neo_alpha 0.5" ;;
  nsmt_pop)       CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init $PSI" ;;
  spk_ssa)        CF="--model myModel --hippo_ssa --hippo_T 4" ;;
  mix_flif_bidir) CF="--model neoMixer --mixer flif --mixer_bidir --neo_alpha 0.5" ;;
  mix_pop_bidir)  CF="--model neoMixer --mixer pop_flif --mixer_bidir --neo_alpha 0.5 --pop_soma_init $PSI" ;;
  mix_lif_bidir)  CF="--model neoMixer --mixer lif --mixer_bidir" ;;
  mix_none)       CF="--model neoMixer --mixer none" ;;
  bix_lif|bixch_lif|bix_flif|bixch_flif|bix_pop_lif|bixch_pop_lif|bix_pop_flif|bixch_pop_flif|bix_popg_lif|bix_popg_flif|bix_popk_lif|bix_popk_flif)
                  case $COND in bixch_*) CB=xattn_ch; NK=${COND#bixch_} ;; *) CB=xattn; NK=${COND#bix_} ;; esac
                  case $NK in
                    lif)       CF="--model neoMixer --mixer lif" ;;
                    flif)      CF="--model neoMixer --mixer flif --neo_alpha 0.5" ;;
                    pop_lif)   CF="--model neoMixer --mixer pop_lif --pop_soma_init 0.5" ;;
                    pop_flif)  CF="--model neoMixer --mixer pop_flif --neo_alpha 0.5 --pop_soma_init 1.0" ;;
                    popg_lif)  CF="--model neoMixer --mixer popg_lif" ;;
                    popg_flif) CF="--model neoMixer --mixer popg_flif --neo_alpha 0.5" ;;
                    popk_lif)  CF="--model streamMixer --mixer popk_lif" ;;
                    popk_flif) CF="--model streamMixer --mixer popk_flif --neo_alpha 0.5" ;;
                  esac
                  CF="$CF --mixer_bidir --mixer_comb $CB" ;;
  mix_pop_lif_bidir)   CF="--model neoMixer --mixer pop_lif --mixer_bidir --pop_soma_init 0.5" ;;
  mix_popg_lif_bidir)  CF="--model neoMixer --mixer popg_lif --mixer_bidir" ;;
  mix_popg_flif_bidir) CF="--model neoMixer --mixer popg_flif --mixer_bidir --neo_alpha 0.5" ;;
  mix_popk_lif_bidir)  CF="--model streamMixer --mixer popk_lif --mixer_bidir" ;;
  mix_popk_flif_bidir) CF="--model streamMixer --mixer popk_flif --mixer_bidir --neo_alpha 0.5" ;;
  spk_ssa_abn)    CF="--model myModel --hippo_ssa --hippo_T 4 --attn_bn" ;;
  nsmt_pop_pe)    CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --pos_emb" ;;
  mix_pop_bidir_pe) CF="--model neoMixer --mixer pop_flif --mixer_bidir --neo_alpha 0.5 --pop_soma_init 1.0 --pos_emb" ;;
  mix_pop_rec_bidir) CF="--model neoMixer --mixer pop_rec --mixer_bidir --neo_alpha 0.5 --pop_soma_init 1.0" ;;
  spk_ssa_nope)   CF="--model myModel --hippo_ssa --hippo_T 4 --ssa_pe 0" ;;
  nsmt_lif_pe)    CF="--model myModel --neo_neuron lif --neo_where both --pos_emb" ;;
  mix_lif_bidir_pe) CF="--model neoMixer --mixer lif --mixer_bidir --pos_emb" ;;
  mix_lif_rec_bidir) CF="--model neoMixer --mixer lif_rec --mixer_bidir" ;;
  mix_sjlif_bidir) CF="--model neoMixer --mixer sj_lif --mixer_bidir" ;;
  mix_pop_bidir_pt16) v=PT16_PSI_$DATA; CF="--model neoMixer --mixer pop_flif --mixer_bidir --neo_alpha 0.5 --pop_taus 16,32,64,128 --pop_soma_init ${!v}" ;;
  nsmt_lif_t4)    CF="--model myModel --neo_neuron lif --neo_where both --hippo_T 4 --hippo_xattn" ;;
  nsmt_pop_t4)    CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn" ;;
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
TAGC=${COND}${PSITAG:-}$([ "$CROP" != 0 ] && echo "_crop$CROP")
OUT=$LOG/${DATA}_${TAGC}_f${FOLD}_seed${S}_lr${LR}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train_rhythm.py \
  --data $DATA --seed $S --log_dir "$LOG/$DATA" --gating attn -s -nd 0 --warm_up_epoch 0 \
  -emb 384 -nh 8 --mlp_ratios 1 --layers 1 --time_layers 2 --keep_ratio $KEEP --alpha $ALPHA --tau 2.0 --no-bias \
  --scheduler cosine -bs ${BS:-32} --eval_bs ${BS:-32} -lr $LR --weight_decay 1e-4 --max_lr 0.005 $HP $CF --tag $TAGC lr$LR seed$S > "$OUT" 2>&1)
RC=$?
D=$(ls -td "$LOG/$DATA"/*/*"+${TAGC}+lr${LR}+seed${S}+"*/seed${S}_* 2>/dev/null | head -1)
CSV="$D/log/final+result.csv"; LCSV="$D/log/last/final+result.csv"
if [ $RC -ne 0 ] || [ ! -s "$CSV" ]; then echo "SKIP(rc=$RC, no result) data=$DATA cond=${COND}${PSITAG:-} fold=$FOLD seed=$S lr=$LR crop=$CROP out=$OUT" | tee -a "$RES/raw.txt"; exit 1; fi
VALS=$($PY - "$CSV" "$LCSV" <<'EOF'
import csv, sys
b = next(csv.DictReader(open(sys.argv[1]))); l = next(csv.DictReader(open(sys.argv[2])))
keys = [k for k in b if k not in ('weights',)]
print(' '.join(f'{k}={b[k]}' for k in keys) + ' ' + ' '.join(f'last_{k}={l[k]}' for k in ('score', 'acc', 'macro_f1')))
EOF
)
BE=$(cat "$D/log/best_epoch.txt" 2>/dev/null | tr ' ' ':')
PAR=$(grep -a 'number of parameters' "$OUT" | grep -oE 'parameters : [0-9]+' | grep -oE '[0-9]+'); t1=$(date +%s)
echo "RESULT exp=nsmt_flif_v1_rhythm data=$DATA cond=${COND}${PSITAG:-} fold=$FOLD seed=$S lr=$LR crop=$CROP $VALS best_epoch_val=$BE params=$PAR time_s=$((t1-t0)) sha=$SHA dir=$D" | tee -a "$RES/raw.txt"
