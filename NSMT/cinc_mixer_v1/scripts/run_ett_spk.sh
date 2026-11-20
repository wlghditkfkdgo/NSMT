#!/usr/bin/env bash
# run_ett_spk.sh (2026-10-31, branch exp/f-lif-pop-v3): one ETT forecasting run of the SpkMixer (spk_mixer.py) with the
# forecaster of forecast_wrap.py / train_forecast.py (copied from exp/nsmt-flif-neo dcfdc4811, ETT_PROVENANCE.md). Built from that
# branch's nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh (blob 8b0a3492): the same training line, per-dataset values (the
# paper's Table 13: patch, emb, heads, mlp, M = mixer layers per direction, lr, batch, epochs), patience 10, best-validation
# weights tested, plain L1 loss. Only ETTh2 / ETTm2 (user decision 12 of 10-14: ETTh2 + ETTm2 x H96 / H720).
# usage: COND=<cond> run_ett_spk.sh <DS: ETTh2|ETTm2> <PL> <seed> <gpu>
#   COND: spkmx_lif_<comb>_<cf>  (SpkMixer, LIF in every layer; comb or|sdor|sdlin|muxor|muxlin, cf iand|or|add)
#         spk_ssa                (Spikformer-style; only for the reproduction check t4 -- the baseline values are reused)
#   env:  SUITE EPOCHS PATIENCE (10) RES LOG PY THREADS
#         LR (0.0005) LAYERS (mixer layers per direction; ETTh2 1, ETTm2 2) -- 2026-11-01 tuning (Table 13 values by default; the
#         training arguments are then the same as before, only the tag / folder name and the RESULT fields tag= lr= layers= are new)
#         SEQ (96) input length -- 2026-11-17 input-length test
#         PATCH (8) EMB (64) MLPR (ETTh2 0.5, ETTm2 1) BLOCKS (1) WD (0.06) -- 2026-11-19 ETT tuning of the LIF SpkMixer (PROJECT_LOG step-2
#         pre-registration): --patch_size, -emb, --mlp_ratios, --layers (blocks), --weight_decay; only the listed values are accepted, a
#         value equal to the default counts as unset (the same training line and tag), any other value adds _pt<v> / _e<v> / _mlp<v> /
#         _blk<v> / _wd<v> to the tag; LIF SpkMixer conditions only (population models would need a gain table per shape)
#         2026-11-20 extension (PROJECT_LOG, user decisions 11-20): also PATCH 24 | 32 and WD 0.005 | 0.02 | 0.05 (tags _pt24, _wd0.02, ...);
#         every earlier value keeps the same training line and tag
set -u
DS=${1:?}; PL=${2:?}; S=${3:?}; GPU=${4:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # cinc_mixer_v1: model code, train_forecast.py, results/, log/
R="$(cd "$W/../.." && pwd)"                            # worktree root
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
COND=${COND:?}
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
# 2026-11-17 input-length test (PROJECT_LOG): SEQ=<input length> -> --seq_len (default 96: the training line is the same text as before), tag _s<SEQ>;
# a population model at SEQ != 96 takes the table made at that length, calib/len/gains_<ds>_pop_lif_muxlin_<connect>_s<SEQ>.json (calibrate_len.py; band none, rep only)
SEQ=${SEQ:-96}; STAG=""; [ "$SEQ" != 96 ] && STAG="_s$SEQ"
case $SEQ in ''|*[!0-9]*) echo "bad SEQ $SEQ"; exit 1 ;; esac
[ "${MIXMS:-0}${MIXRESET:-spikede}" != 0spikede ] && case $COND in spkmx_pop_*|spkmx_flif_*) echo "MIXMS / MIXRESET: LIF SpkMixer only"; exit 1 ;; esac
# 2026-11-18 neuron placement test (PROJECT_LOG): NSCOPE=mixer -> --spk_neuron_scope mixer (tag _nsmixer): population only in the directional
# stacks + fusion, embedding / MLP the single neuron of the family; population models with NSCOPE=mixer, MIXPOOL or pop_flif take the table
# calib/pn/gains_<task>_<mixer>_muxlin_<connect>_sc<scope>_mp<pool>[_s<SEQ>].json (calibrate_pn.py); pop_lif / all / none keeps its earlier table
NS=${NSCOPE:-all}
case $NS in all) ;; mixer) case $COND in spkmx_pop_*) BX="$BX --spk_neuron_scope mixer"; BTAG="${BTAG}_nsmixer" ;; *) echo "NSCOPE=mixer is for population conditions"; exit 1 ;; esac ;; *) echo "bad NSCOPE"; exit 1 ;; esac
case $COND in spkmx_pop_*)
  PMX=pop_lif; case $COND in spkmx_pop_flif_*) PMX=pop_flif ;; esac
  GT="$W/calib/stft/gains_${DS}_pop_lif_muxlin_${CFA}_bd${BAND:-none}_tx${TAXIS:-rep}.json"
  if [ "$PMX" = pop_flif ] || [ "$NS" != all ] || [ "${MIXPOOL:-none}" != none ]; then
    [ "${BAND:-none}" = none ] && [ "${TAXIS:-rep}" = rep ] || { echo "NSCOPE / MIXPOOL / pop_flif: band none and rep only"; exit 1; }
    GT="$W/calib/pn/gains_${DS}_${PMX}_muxlin_${CFA}_sc${NS}_mp${MIXPOOL:-none}${STAG}.json"
  elif [ "$SEQ" != 96 ]; then
    [ "${BAND:-none}" = none ] && [ "${TAXIS:-rep}" = rep ] || { echo "SEQ != 96 with a population model: band none and rep only"; exit 1; }
    GT="$W/calib/len/gains_${DS}_pop_lif_muxlin_${CFA}_s${SEQ}.json"
  fi
  [ -s "$GT" ] || { echo "missing gain table $GT"; exit 1; }
  BX="$BX --gain_table $GT" ;;
esac
case $DS in                       # patch emb heads mlp M lr bs epochs   (as run_ett_mixer.sh; heads unused by the SpkMixer)
  ETTh2) L0=1; E=100 ;;           # Table 13: --patch_size 8 -emb 64 -nh 8  --mlp_ratios 0.5 --time_layers 1 -lr 0.0005 -bs 64
  ETTm2) L0=2; E=50 ;;            # Table 13: --patch_size 8 -emb 64 -nh 16 --mlp_ratios 1   --time_layers 2 -lr 0.0005 -bs 64
  *) echo "bad dataset $DS (ETTh2 / ETTm2 only)"; exit 1 ;;
esac
LR=${LR:-0.0005}; LAYERS=${LAYERS:-$L0}                                  # 2026-11-01 tuning knobs
case $DS in ETTh2) MLP0=0.5 ;; ETTm2) MLP0=1 ;; esac                      # 2026-11-19 ETT tuning knobs (defaults = Table 13 values)
PATCH=${PATCH:-8}; EMB=${EMB:-64}; MLPR=${MLPR:-$MLP0}; BLOCKS=${BLOCKS:-1}; WD=${WD:-0.06}
case $PATCH in 4|8|16|24|32) ;; *) echo "bad PATCH $PATCH (4|8|16|24|32)"; exit 1 ;; esac
case $EMB in 32|64|128|256) ;; *) echo "bad EMB $EMB (32|64|128|256)"; exit 1 ;; esac
case $MLPR in 0.5|1|2|4) ;; *) echo "bad MLPR $MLPR (0.5|1|2|4)"; exit 1 ;; esac
case $BLOCKS in 1|2|3) ;; *) echo "bad BLOCKS $BLOCKS (1|2|3)"; exit 1 ;; esac
case $WD in 0|0.005|0.01|0.02|0.05|0.06|0.1) ;; *) echo "bad WD $WD (0|0.005|0.01|0.02|0.05|0.06|0.1)"; exit 1 ;; esac
KTAG=""
[ "$PATCH" != 8 ] && KTAG="${KTAG}_pt$PATCH"
[ "$EMB" != 64 ] && KTAG="${KTAG}_e$EMB"
[ "$MLPR" != "$MLP0" ] && KTAG="${KTAG}_mlp$MLPR"
[ "$BLOCKS" != 1 ] && KTAG="${KTAG}_blk$BLOCKS"
[ "$WD" != 0.06 ] && KTAG="${KTAG}_wd$WD"
[ -n "$KTAG" ] && case $COND in spkmx_lif_*) ;; *) echo "PATCH / EMB / MLPR / BLOCKS / WD: LIF SpkMixer conditions only"; exit 1 ;; esac
case $DS in
  ETTh2) HP="--patch_size $PATCH -emb $EMB -nh 8  --mlp_ratios $MLPR --time_layers $LAYERS -lr $LR -bs 64" ;;
  ETTm2) HP="--patch_size $PATCH -emb $EMB -nh 16 --mlp_ratios $MLPR   --time_layers $LAYERS -lr $LR -bs 64" ;;
esac
TAGC="${COND}_lr${LR}_L${LAYERS}${BTAG}${STAG}${KTAG}"
HM=""; [ "${HEADMEM:-0}" = 1 ] && { HM="--head_mem"; TAGC="${TAGC}_hm"; }   # 2026-11-03 H1 (head_mem.py): HEADMEM=1 -> --head_mem, tag _hm
[ "${MS:-0}" = 1 ] && { [ "$COND" = spk_ssa ] || { echo "MS=1 is for spk_ssa (the SpkMixer uses cond spkmx_lif_<comb>_ms)"; exit 1; }; HM="$HM --ms_shortcut"; TAGC="${TAGC}_ms"; }   # 2026-11-08 membrane shortcut
case ${NALL:-sj_lif} in sj_lif) ;; lif) { [ "$COND" = spk_ssa ] && [ "${MS:-0}" != 1 ]; } || { echo "NALL=lif is for spk_ssa without MS"; exit 1; }
  HM="$HM --neuron_all lif"; TAGC="${TAGC}_nalif" ;; *) echo "bad NALL"; exit 1 ;; esac   # 2026-11-18: Spikformer-style with the spikeDE LIF (neuron_all)
TL="";[ "${LOSS:-l1}" = mse ] && { TL="--train_loss mse"; TAGC="${TAGC}_mse"; }   # 2026-11-07 L2 test: LOSS=mse -> --train_loss mse, tag _mse
E=${EPOCHS:-$E}; P=${PATIENCE:-10}
SUITE=${SUITE:-flif-v1-ett-spkmux-$(date +%Y%m%d)}
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}
mkdir -p "$RES" "$LOG"; RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
OUT=$LOG/ett_${DS}_p${PL}_${TAGC}_seed${S}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train_forecast.py \
  --log_dir "$LOG/$DS/p$PL" --seed $S -s -nd 0 --gating attn --no-bias --scheduler reduce -e $E --warm_up_epoch 0 \
  --layers $BLOCKS --tau 2.0 --weight_decay $WD --patience $P \
  --data $DS --data_path $DS.csv --root_path "$W/../forecasting/dataset/ETT-small" --seq_len $SEQ --pred_len $PL --features M --target OT \
  $HP $CF $BX $HM $TL --tag $TAGC seed$S > "$OUT" 2>&1)
RC=$?
D=$(ls -td "$LOG/$DS/p$PL"/*/*"+${TAGC}+seed${S}+"*"+epoch+${E}+"*/seed${S}_* 2>/dev/null | head -1)
CSV="$D/log/final+result.csv"; LCSV="$D/log/last/final+result.csv"
if [ $RC -ne 0 ] || [ ! -s "$CSV" ]; then echo "SKIP(rc=$RC, no result) ds=$DS pl=$PL cond=$COND seed=$S out=$OUT" | tee -a "$RES/raw.txt"; exit 1; fi
VALS=$($PY - "$CSV" "$LCSV" <<'EOF'
import csv, sys
b = next(csv.DictReader(open(sys.argv[1]))); l = next(csv.DictReader(open(sys.argv[2])))
keys = [k for k in b if k not in ('weights',)]
print(' '.join(f'{k}={b[k]}' for k in keys) + ' ' + ' '.join(f'last_{k}={l[k]}' for k in ('mse', 'mae')))
EOF
)
[ -z "$VALS" ] && { echo "SKIP(result parse failed) ds=$DS pl=$PL cond=$COND seed=$S out=$OUT" | tee -a "$RES/raw.txt"; exit 1; }
BE=$(cat "$D/log/best_epoch.txt" 2>/dev/null | tr ' ' ':')
PAR=$(grep -a 'number of parameters' "$OUT" | grep -oE 'parameters : [0-9]+' | grep -oE '[0-9]+'); t1=$(date +%s)
echo "RESULT exp=cinc_mixer_v1_ett_spkmux ds=$DS pl=$PL cond=$COND tag=$TAGC lr=$LR layers=$LAYERS patch=$PATCH emb=$EMB mlpr=$MLPR blocks=$BLOCKS wd=$WD seed=$S $VALS best_epoch_val_epochs=$BE params=$PAR time_s=$((t1-t0)) sha=$SHA dir=$D" | tee -a "$RES/raw.txt"
