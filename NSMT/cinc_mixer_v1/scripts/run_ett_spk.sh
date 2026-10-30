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
    case $CFX in iand) CFA=IAND ;; or) CFA=OR ;; add) CFA=ADD ;; *) echo "bad connect in $COND"; exit 1 ;; esac
    [ "$REST" = "spkmx_lif_$COMB" ] || { echo "bad cond $COND"; exit 1; }
    CF="--model spkMixer --mixer lif --mixer_comb $COMB --connect_f $CFA --hippo_T 4" ;;
  *) echo "bad cond $COND"; exit 1 ;;
esac
case $DS in                       # patch emb heads mlp M lr bs epochs   (as run_ett_mixer.sh; heads unused by the SpkMixer)
  ETTh2) HP="--patch_size 8 -emb 64 -nh 8  --mlp_ratios 0.5 --time_layers 1 -lr 0.0005 -bs 64"; E=100 ;;
  ETTm2) HP="--patch_size 8 -emb 64 -nh 16 --mlp_ratios 1   --time_layers 2 -lr 0.0005 -bs 64"; E=50 ;;
  *) echo "bad dataset $DS (ETTh2 / ETTm2 only)"; exit 1 ;;
esac
E=${EPOCHS:-$E}; P=${PATIENCE:-10}
SUITE=${SUITE:-flif-v1-ett-spkmux-$(date +%Y%m%d)}
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}
mkdir -p "$RES" "$LOG"; RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
OUT=$LOG/ett_${DS}_p${PL}_${COND}_seed${S}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train_forecast.py \
  --log_dir "$LOG/$DS/p$PL" --seed $S -s -nd 0 --gating attn --no-bias --scheduler reduce -e $E --warm_up_epoch 0 \
  --layers 1 --tau 2.0 --weight_decay 0.06 --patience $P \
  --data $DS --data_path $DS.csv --root_path "$W/../forecasting/dataset/ETT-small" --seq_len 96 --pred_len $PL --features M --target OT \
  $HP $CF --tag $COND seed$S > "$OUT" 2>&1)
RC=$?
D=$(ls -td "$LOG/$DS/p$PL"/*/*"+${COND}+seed${S}+"*"+epoch+${E}+"*/seed${S}_* 2>/dev/null | head -1)
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
echo "RESULT exp=cinc_mixer_v1_ett_spkmux ds=$DS pl=$PL cond=$COND seed=$S $VALS best_epoch_val_epochs=$BE params=$PAR time_s=$((t1-t0)) sha=$SHA dir=$D" | tee -a "$RES/raw.txt"
