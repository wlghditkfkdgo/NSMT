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
    case $CFX in iand) CFA=IAND ;; or) CFA=OR ;; add) CFA=ADD ;; *) echo "bad connect in $COND"; exit 1 ;; esac
    [ "$REST" = "spkmx_lif_$COMB" ] || { echo "bad cond $COND"; exit 1; }
    CF="--model spkMixer --mixer lif --mixer_comb $COMB --connect_f $CFA --hippo_T 4" ;;
  *) echo "bad cond $COND"; exit 1 ;;
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
VF=${VAL_FOLD:-}; VFA=""; TAGC="${COND}_lr${LR}_L${ML}"
[ -n "$VF" ] && { VFA="--val_fold $VF"; TAGC="${TAGC}_v${VF}"; }
HM=""; [ "${HEADMEM:-0}" = 1 ] && { HM="--head_mem"; TAGC="${TAGC}_hm"; }   # 2026-11-03 H1 (head_mem.py): HEADMEM=1 -> --head_mem, tag _hm
[ -n "${HEADPOOL:-}" ] && [ "$HEADPOOL" != none ] && { HM="$HM --head_pool $HEADPOOL"; TAGC="${TAGC}_hp${HEADPOOL}"; }   # 2026-11-03 head input forms (head_pool.py)
OUT=$LOG/mitbih_${TAGC}_seed${S}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train.py \
  --model myModel --seed $S --log_dir "$LOG/MITBIH" --gating attn --attn SSA -s --test --print_epoch 1 -nd 0 \
  -e $EPOCHS --warm_up_epoch 0 --num_channels 1 -bs 512 -lr $LR --max_lr 0.005 -emb 384 -nh 8 --mlp_ratios 1 \
  --layers $LAYERS --keep_ratio 0.15 --time_layers $ML --alpha 0.1 --patch_size 32 --no-bias --scheduler cosine \
  --data UEA --data_path ECG --seq_len 187 --root_path "$DATA" $CF $VFA $HM --tag $TAGC seed$S > "$OUT" 2>&1)
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
