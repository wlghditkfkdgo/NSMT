#!/usr/bin/env bash
# run_ad_spk.sh (2026-11-02, branch exp/f-lif-pop-v3; PROJECT_LOG AD pre-registration): one anomaly-detection run of the SpkMixer
# or the Spikformer-style model with ad_wrap.Reconstructor / train_ad.py, TSLib protocol (window 100, step 1, MSE reconstruction,
# threshold percentile = anomaly_ratio 1 for MSL / SMAP / PSM as neorecall_ad_v1/anomaly_detection/scripts/run_recall_ad.sh).
# Body values = the ETT runs (run_ett_spk.sh: patch 8, emb 64, mlp 1, weight decay 0.06, tau 2); batch 128 (TSLib AD); heads 8.
# usage: COND=<cond> run_ad_spk.sh <DS: MSL|SMAP|PSM> <seed> <gpu>
#   COND: spkmx_lif_<comb>_<cf>  (SpkMixer, LIF in every layer; comb sdlin|muxlin|..., cf iand|or|add)
#         spk_ssa                (Spikformer-style)
#   env:  SUITE EPOCHS (10) PATIENCE (3) RES LOG PY THREADS DATA_ROOT
#         LR (0.0005) MLAYERS (mixer layers per direction, 2) HOLDOUT (0 | 1: tuning run, first 80 % / last 20 %, no test)
set -u
DS=${1:?}; S=${2:?}; GPU=${3:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # cinc_mixer_v1
R="$(cd "$W/../.." && pwd)"                            # worktree root
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
DATA_ROOT=${DATA_ROOT:-/home/yschoi/CLS_spiking_transformer/Bio-inspired-Spiking-Memory-Transformer-for-time-series-representation-learning/NSMT/anomaly_detection/dataset}
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
case $DS in MSL|SMAP|PSM) AR=1 ;; *) echo "bad dataset $DS (MSL / SMAP / PSM)"; exit 1 ;; esac
LR=${LR:-0.0005}; ML=${MLAYERS:-2}; HO=${HOLDOUT:-0}
case $HO in 0) HOA="" ; HOT="" ;; 1) HOA="--holdout"; HOT="_ho" ;; *) echo "HOLDOUT 0 or 1"; exit 1 ;; esac
TAGC="${COND}_lr${LR}_L${ML}${HOT}"
E=${EPOCHS:-10}; P=${PATIENCE:-3}
SUITE=${SUITE:-flif-v1-ad-spkmux-$(date +%Y%m%d)}
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}
mkdir -p "$RES" "$LOG"; RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
OUT=$LOG/ad_${DS}_${TAGC}_seed${S}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train_ad.py \
  --log_dir "$LOG/$DS" --seed $S -s -nd 0 --gating attn --no-bias --scheduler reduce -e $E --warm_up_epoch 0 \
  --layers 1 --tau 2.0 --weight_decay 0.06 --patience $P --anomaly_ratio $AR $HOA \
  --data $DS --root_path "$DATA_ROOT/$DS" --seq_len 100 \
  --patch_size 8 -emb 64 -nh 8 --mlp_ratios 1 --time_layers $ML -lr $LR -bs 128 \
  $CF --tag $TAGC seed$S > "$OUT" 2>&1)
RC=$?
D=$(ls -td "$LOG/$DS"/*/*"+${TAGC}+seed${S}+"*"+epoch+${E}+"*/seed${S}_* 2>/dev/null | head -1)
CSV="$D/log/final+result.csv"
if [ $RC -ne 0 ] || [ ! -s "$CSV" ]; then echo "SKIP(rc=$RC, no result) ds=$DS cond=$COND tag=$TAGC seed=$S out=$OUT" | tee -a "$RES/raw.txt"; exit 1; fi
VALS=$($PY - "$CSV" <<'EOF'
import csv, sys
b = next(csv.DictReader(open(sys.argv[1])))
print(' '.join(f'{k}={v}' for k, v in b.items() if k != 'weights'))
EOF
)
[ -z "$VALS" ] && { echo "SKIP(result parse failed) ds=$DS cond=$COND tag=$TAGC seed=$S out=$OUT" | tee -a "$RES/raw.txt"; exit 1; }
PAR=$(grep -a 'number of parameters' "$OUT" | grep -oE 'parameters : [0-9]+' | grep -oE '[0-9]+'); t1=$(date +%s)
echo "RESULT exp=cinc_mixer_v1_ad_spkmux ds=$DS cond=$COND tag=$TAGC lr=$LR layers=$ML holdout=$HO seed=$S $VALS params=$PAR time_s=$((t1-t0)) sha=$SHA dir=$D" | tee -a "$RES/raw.txt"
