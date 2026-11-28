#!/usr/bin/env bash
# run_mitbih_spk.sh (2026-11-01, branch exp/f-lif-pop-v3): one MIT-BIH run of the SpkMixer (spk_mixer.py) with the ECG framework
# entry train.py of this folder (byte-identical to the one behind the other session's MIT-BIH baseline runs). Built from
# exp/nsmt-flif-neo dcfdc4811 NSMT/nsmt_flif_v1/classification/scripts/run_mitbih2.sh (blob 0d726ba1; ETT_PROVENANCE.md):
# the same training line (30 epochs, cosine, bs 512, lr 0.01, emb 384, patch 32 = 11 tokens, mixer layers 2, CAL-weighted CE;
# the last-epoch weights are tested on the 30 % test file, nothing is selected) and RESULT line; only the conditions differ.
# usage: COND=<cond> run_mitbih_spk.sh <seed> <gpu>
#   COND: spkmx_lif_<comb>_<cf>  (SpkMixer, LIF in every layer; comb or|sdor|sdlin|muxor|muxlin, cf iand|or|add)
#         spk_ssa                (Spikformer-style; only for the reproduction check -- the baseline values are reused)
#   env:  SUITE EPOCHS (30) RES LOG DATA PY THREADS
#         LR (0.01) MLAYERS (mixer layers per direction, --time_layers; 2) VAL_FOLD (unset = no validation split; k = stratified fold
#         k of 5 of the training file as validation, train.py --val_fold) -- 2026-11-01 tuning; defaults = the original arguments
#         TMEM (on | off) -- 2026-11-28 memoryless control: off -> --token_memory off, tag _tmoff; unset = the original arguments
set -u
S=${1:?}; GPU=${2:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # cinc_mixer_v1
R="$(cd "$W/../.." && pwd)"                            # worktree root
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
COND=${COND:?}; ALPHA=0.5
case $COND in
  spk_ssa) CF="--model myModel --hippo_ssa --hippo_T 4" ;;
  spkmx_lif_*)
    CFX=${COND##*_}; REST=${COND%_*}; COMB=${REST##*_}
    case $COMB in or|sdor|sdlin|muxor|muxlin) ;; *) echo "bad comb in $COND"; exit 1 ;; esac
    case $CFX in iand) CFA=IAND ;; or) CFA=OR ;; add) CFA=ADD ;; ms) CFA=MS ;; *) echo "bad connect in $COND"; exit 1 ;; esac
    [ "$REST" = "spkmx_lif_$COMB" ] || { echo "bad cond $COND"; exit 1; }
    CF="--model spkMixer --mixer lif --mixer_comb $COMB --connect_f $CFA --hippo_T 4" ;;
  spkmx_pop_lif_muxlin_*)                                # 2026-11-16 STFT band test: population LIF in every layer (soma init 0.5, gain table below)
    CFX=${COND##*_}; COMB=muxlin
    case $CFX in iand) CFA=IAND ;; or) CFA=OR ;; add) CFA=ADD ;; *) echo "bad connect in $COND"; exit 1 ;; esac
    CF="--model spkMixer --mixer pop_lif --mixer_comb $COMB --connect_f $CFA --hippo_T 4 --pop_soma_init 0.5" ;;
  spkmx_flif_muxlin_*)                                   # 2026-11-18 neuron placement test: f-LIF (spikeDE, alpha 0.5) in every layer
    CFX=${COND##*_}; COMB=muxlin
    case $CFX in iand) CFA=IAND ;; or) CFA=OR ;; add) CFA=ADD ;; *) echo "bad connect in $COND"; exit 1 ;; esac
    CF="--model spkMixer --mixer flif --mixer_comb $COMB --connect_f $CFA --hippo_T 4" ;;
  spkmx_pop_flif_muxlin_*)                               # 2026-11-18: population f-LIF (soma init 1.0 as every population f-LIF run; gain table below)
    CFX=${COND##*_}; COMB=muxlin
    case $CFX in iand) CFA=IAND ;; or) CFA=OR ;; add) CFA=ADD ;; *) echo "bad connect in $COND"; exit 1 ;; esac
    CF="--model spkMixer --mixer pop_flif --mixer_comb $COMB --connect_f $CFA --hippo_T 4 --pop_soma_init 1.0" ;;
  *) echo "bad cond $COND"; exit 1 ;;
esac
# 2026-11-16 STFT band test (PROJECT_LOG): BAND=group|member -> --spk_band, TAXIS=token -> --spk_taxis token (tags _bd<band>, _tx<taxis>);
# a population model always takes its own gain table calib/stft/gains_<task>_pop_lif_muxlin_<connect>_bd<band>_tx<taxis>.json (calibrate_stft.py)
BX=""; BTAG=""
case ${BAND:-none} in none) ;; group|member) BX="--spk_band $BAND"; BTAG="_bd$BAND" ;; *) echo "bad BAND"; exit 1 ;; esac
case ${TAXIS:-rep} in rep) ;; token) BX="$BX --spk_taxis token"; BTAG="${BTAG}_txtoken" ;; *) echo "bad TAXIS"; exit 1 ;; esac
# 2026-11-17 mixer variants (PROJECT_LOG): MIXPOOL=max|mean -> --spk_mixpool (tag _mp<x>), MIXMS=1 -> --mixer_ms (tag _mms),
# MIXRESET=subtract|zero -> --mixer_reset (tag _mr<x>); SpkMixer only; population models need their own gain table (not yet)
case ${MIXPOOL:-none} in none) ;; max|mean) BX="$BX --spk_mixpool $MIXPOOL"; BTAG="${BTAG}_mp$MIXPOOL" ;; *) echo "bad MIXPOOL"; exit 1 ;; esac
case ${MIXMS:-0} in 0) ;; 1) BX="$BX --mixer_ms"; BTAG="${BTAG}_mms" ;; *) echo "bad MIXMS"; exit 1 ;; esac
case ${MIXRESET:-spikede} in spikede) ;; subtract|zero) BX="$BX --mixer_reset $MIXRESET"; BTAG="${BTAG}_mr$MIXRESET" ;; *) echo "bad MIXRESET"; exit 1 ;; esac
[ -n "$BX" ] && case $COND in spkmx_*) ;; *) echo "BAND / TAXIS / MIXPOOL / MIXMS / MIXRESET are for the SpkMixer"; exit 1 ;; esac
[ "${MIXMS:-0}${MIXRESET:-spikede}" != 0spikede ] && case $COND in spkmx_pop_*|spkmx_flif_*) echo "MIXMS / MIXRESET: LIF SpkMixer only"; exit 1 ;; esac
# 2026-11-18 neuron placement test (PROJECT_LOG): NSCOPE=mixer -> --spk_neuron_scope mixer (tag _nsmixer): population only in the directional
# stacks + fusion, embedding / MLP the single neuron of the family; population models with NSCOPE=mixer, MIXPOOL or pop_flif take the table
# calib/pn/gains_<task>_<mixer>_muxlin_<connect>_sc<scope>_mp<pool>[_s<SEQ>].json (calibrate_pn.py); pop_lif / all / none keeps its earlier table
NS=${NSCOPE:-all}
case $NS in all) ;; mixer) case $COND in spkmx_pop_*) BX="$BX --spk_neuron_scope mixer"; BTAG="${BTAG}_nsmixer" ;; *) echo "NSCOPE=mixer is for population conditions"; exit 1 ;; esac ;; *) echo "bad NSCOPE"; exit 1 ;; esac
case $COND in spkmx_pop_*)
  PMX=pop_lif; case $COND in spkmx_pop_flif_*) PMX=pop_flif ;; esac
  GT="$W/calib/stft/gains_MITBIH_pop_lif_muxlin_${CFA}_bd${BAND:-none}_tx${TAXIS:-rep}.json"
  if [ "$PMX" = pop_flif ] || [ "$NS" != all ] || [ "${MIXPOOL:-none}" != none ]; then
    [ "${BAND:-none}" = none ] && [ "${TAXIS:-rep}" = rep ] || { echo "NSCOPE / MIXPOOL / pop_flif: band none and rep only"; exit 1; }
    GT="$W/calib/pn/gains_MITBIH_${PMX}_muxlin_${CFA}_sc${NS}_mp${MIXPOOL:-none}.json"
  fi
  [ -s "$GT" ] || { echo "missing gain table $GT"; exit 1; }
  BX="$BX --gain_table $GT" ;;
esac
LAYERS=1                                                # model blocks (--layers), as run_mitbih2.sh
LR=${LR:-0.01}; ML=${MLAYERS:-2}                      # MLAYERS = mixer layers per direction (--time_layers)
SUITE=${SUITE:-flif-v1-mitbih-spkmux-$(date +%Y%m%d)}
EPOCHS=${EPOCHS:-30}
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}
DATA=${DATA:-/home/yschoi/CLS_spiking_transformer/data}   # 5class_mitbih_{train_7,test_3}_187.csv (read only; md5 = the baseline's copy)
mkdir -p "$RES" "$LOG"
RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
VF=${VAL_FOLD:-}; VFA=""; TAGC="${COND}_lr${LR}_L${ML}${BTAG}"
[ -n "$VF" ] && { VFA="--val_fold $VF"; TAGC="${TAGC}_v${VF}"; }
HM=""; [ "${HEADMEM:-0}" = 1 ] && { HM="--head_mem"; TAGC="${TAGC}_hm"; }   # 2026-11-03 H1 (head_mem.py): HEADMEM=1 -> --head_mem, tag _hm
[ -n "${HEADPOOL:-}" ] && [ "$HEADPOOL" != none ] && { HM="$HM --head_pool $HEADPOOL"; TAGC="${TAGC}_hp${HEADPOOL}"; }   # 2026-11-03 head input forms (head_pool.py)
[ "${MS:-0}" = 1 ] && { [ "$COND" = spk_ssa ] || { echo "MS=1 is for spk_ssa (the SpkMixer uses cond spkmx_lif_<comb>_ms)"; exit 1; }; HM="$HM --ms_shortcut"; TAGC="${TAGC}_ms"; }   # 2026-11-08 membrane shortcut
case ${NALL:-sj_lif} in sj_lif) ;; lif) { [ "$COND" = spk_ssa ] && [ "${MS:-0}" != 1 ]; } || { echo "NALL=lif is for spk_ssa without MS"; exit 1; }
  HM="$HM --neuron_all lif"; TAGC="${TAGC}_nalif" ;; *) echo "bad NALL"; exit 1 ;; esac   # 2026-11-18: Spikformer-style with the spikeDE LIF (neuron_all)
case ${TMEM:-on} in on) ;; off) case $COND in spkmx_lif_*) HM="$HM --token_memory off"; TAGC="${TAGC}_tmoff" ;; *) echo "TMEM: LIF SpkMixer only"; exit 1 ;; esac ;;
  *) echo "bad TMEM"; exit 1 ;; esac   # 2026-11-28 memoryless control (user "분류 과제 통제"): every token-axis mixer neuron resets after each token
OUT=$LOG/mitbih_${TAGC}_seed${S}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train.py \
  --model myModel --seed $S --log_dir "$LOG/MITBIH" --gating attn --attn SSA -s --test --print_epoch 1 -nd 0 \
  -e $EPOCHS --warm_up_epoch 0 --num_channels 1 -bs 512 -lr $LR --max_lr 0.005 -emb 384 -nh 8 --mlp_ratios 1 \
  --layers $LAYERS --keep_ratio 0.15 --time_layers $ML --alpha 0.1 --patch_size 32 --no-bias --scheduler cosine \
  --data UEA --data_path ECG --seq_len 187 --root_path "$DATA" $CF $BX $VFA $HM --tag $TAGC seed$S > "$OUT" 2>&1)
RC=$?
D=$(ls -td "$LOG/MITBIH"/*/*"+${TAGC}+seed${S}+model+"*/seed${S}_* 2>/dev/null | head -1)
CSV="$D/log/final+result.csv"
if [ $RC -ne 0 ] || [ ! -s "$CSV" ]; then echo "SKIP(rc=$RC, no result) cond=$COND seed=$S out=$OUT" | tee -a "$RES/raw.txt"; exit 1; fi
VALS=$($PY - "$CSV" <<'EOF'
import sys
rows = [l.rstrip('\n').split(', ') for l in open(sys.argv[1]) if l.strip()]
h, v = rows[-2], rows[-1]
d = dict(zip(h, v))
keys = ['overall_acc', 'average_acc', 'f1', 'precision', 'sensitivity', 'f1_0', 'f1_1', 'f1_2', 'f1_3', 'f1_4',
        'acc0', 'acc1', 'acc2', 'acc3', 'acc4', 'neuron_lif_uJ', 'neuron_model_fft_uJ', 'neuron_model_direct_uJ',
        'total_fft_uJ', 'total_direct_uJ', 'rho_neo', 'rho_hippo', 'n_test']
out = ' '.join(f'{k}={d[k]}' for k in keys)
out += ' syn_energy_uJ=' + d['energy'].replace(' uJ', '') + ' firing_rate=' + d['firing_rate'].replace(' %', '')
print(out)
EOF
)
[ -z "$VALS" ] && { echo "SKIP(result parse failed) cond=$COND seed=$S out=$OUT" | tee -a "$RES/raw.txt"; exit 1; }
LRS=$(grep -aoE '^epoch +[0-9]+ \(lr=[0-9.e-]+\)' "$OUT" | grep -oE 'lr=[0-9.e-]+' | cut -d= -f2)
EP=$(echo "$LRS" | grep -c .); LR_FINAL=$(echo "$LRS" | tail -1)
PAR=$(grep -a 'number of parameters' "$OUT" | grep -oE '[0-9]+' | tail -1); t1=$(date +%s)
VL=""; if [ -n "$VF" ]; then VC="$D/log/val_last.csv"; [ -s "$VC" ] || { echo "SKIP(no val_last.csv) cond=$COND seed=$S out=$OUT" | tee -a "$RES/raw.txt"; exit 1; }
  VL=" val_f1_last=$(tail -1 "$VC" | cut -d, -f4) val_acc_last=$(tail -1 "$VC" | cut -d, -f5) n_val=$(tail -1 "$VC" | cut -d, -f3)"; fi
echo "RESULT exp=cinc_mixer_v1_mitbih_spkmux cond=$COND tag=$TAGC lr=$LR layers=$ML val_fold=${VF:--1}${VL} dataset=MITBIH seed=$S $VALS epochs=$EPOCHS epochs_run=$EP lr_final=$LR_FINAL params=$PAR time_s=$((t1-t0)) sha=$SHA alpha=$ALPHA dir=$D" | tee -a "$RES/raw.txt"
