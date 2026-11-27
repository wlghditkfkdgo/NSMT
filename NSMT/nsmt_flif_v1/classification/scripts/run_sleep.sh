#!/usr/bin/env bash
# MODI 2026-11-26 ((나), user decisions 11-25 / 11-26): one SNN run on Sleep-EDF-20 with epoch-spectrum tokens (train_rhythm.py,
# data SLEEPEDF = oc_data.build_sleep_psd; the target epoch and the 15 epochs before it, one token per epoch, the head reads the
# target token). Coupling-form comparison: Spikformer(DE) vs the query form and the key / value form, each with pop LIF / pop f-LIF.
# usage (from NSMT/, the pool appends the gpu): SUITE=<suite> LR=<lr> bash nsmt_flif_v1/classification/scripts/run_sleep.sh <cond> <fold> <gpu>
#   cond: spkall_lif | q_pop_lif | q_pop_flif | kv_pop_lif | kv_pop_flif      (seed 42; env SEED, EPOCHS (30), BS (64) for smoke only)
#         + MODI 2026-11-27: q_pop_<lif|flif>_cshuf (context shuffled), q_pop_<lif|flif>_nml (memoryless Neocortex)
#         + MODI 2026-11-27 (user decision 11-27, default neuron set LIF / f-LIF / pop LIF): q_lif | q_flif | kv_lif | kv_flif (plain
#           spikeDE LIF / f-LIF Neocortex, flags as run_rhythm_addneo.sh); the suffixes also apply to q_lif / q_flif
# Writes one RESULT (or SKIP) line to nsmt_flif_v1/classification/results/<suite>/raw.txt.
set -u
COND=$1; FOLD=$2; GPU=$3
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; R="$(cd "$W/../.." && pwd)"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
S=${SEED:-42}; EP=${EPOCHS:-30}; BS=${BS:-64}; LR=${LR:?LR}; SUITE=${SUITE:?SUITE}
Q="--model myModel --neo_where both --hippo_T 4 --hippo_xattn --neo_grad 1.0 --neo_res ms --neuron_all lif"
XF=""; AL=0; BASE=$COND                 # MODI 2026-11-27 ((나) follow-up): suffix _cshuf = context epochs shuffled per sample
while :; do                              # (--ctx_shuffle), _nml = memoryless Neocortex (--neo_memoryless); MODI 2026-11-27 (evening):
  case $BASE in                          # _mem = Neocortex membrane readout (--neo_readout mem), _acc = past-reconstruction aux
    *_cshuf) XF="$XF --ctx_shuffle"; BASE=${BASE%_cshuf} ;;        # (--aux_past rand, alpha 0.1); suffixes combine, query forms only
    *_nml)   XF="$XF --neo_memoryless"; BASE=${BASE%_nml} ;;
    *_acc)   XF="$XF --aux_past rand"; AL=0.1; BASE=${BASE%_acc} ;;
    *_mem)   XF="$XF --neo_readout mem"; BASE=${BASE%_mem} ;;
    *) break ;;
  esac
done
case $BASE in q_pop_lif|q_pop_flif|q_lif|q_flif) ;; *) [ -n "$XF" ] && { echo "suffixes only on q_pop_lif / q_pop_flif / q_lif / q_flif"; exit 2; } ;; esac
case $BASE in
  spkall_lif)  CF="--model myModel --hippo_ssa --hippo_T 4 --neuron_all lif" ;;                                  # Spikformer(DE)
  q_pop_lif)   CF="$Q --neo_neuron pop_lif --pop_soma_init 0.5 --hippo_xattn_mode query" ;;                       # = nsmt4q_pop_lif_msn_sde
  q_pop_flif)  CF="$Q --neo_neuron pop_flif --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_xattn_mode query" ;;      # = nsmt4q_pop_flif_msn_sde
  kv_pop_lif)  CF="$Q --neo_neuron pop_lif --pop_soma_init 0.5" ;;                                                 # key / value form (replace)
  kv_pop_flif) CF="$Q --neo_neuron pop_flif --neo_alpha 0.5 --pop_soma_init 1.0" ;;
  q_lif)       CF="$Q --neo_neuron lif --hippo_xattn_mode query" ;;                                                # MODI 2026-11-27: plain
  q_flif)      CF="$Q --neo_neuron flif --neo_alpha 0.5 --hippo_xattn_mode query" ;;                               # LIF / f-LIF
  kv_lif)      CF="$Q --neo_neuron lif" ;;
  kv_flif)     CF="$Q --neo_neuron flif --neo_alpha 0.5" ;;
  *) echo "bad cond $COND"; exit 2 ;;
esac
CF="$CF $XF"
RES=$W/results/$SUITE; LOG=$W/log/$SUITE; mkdir -p "$RES" "$LOG"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
OUT=$LOG/SLEEPEDF_${COND}_f${FOLD}_seed${S}_lr${LR}_ep${EP}_bs${BS}_${TS}_${SHA}.stdout; t0=$(date +%s)
[ -e "$OUT" ] && { echo "SKIP(stdout exists) cond=$COND fold=$FOLD seed=$S lr=$LR out=$OUT" | tee -a "$RES/raw.txt"; exit 1; }
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train_rhythm.py \
  --data SLEEPEDF --fold $FOLD --ctx_epochs 16 --seed $S --log_dir "$LOG/SLEEPEDF" --gating attn -s -nd 0 --warm_up_epoch 0 \
  -emb 384 -nh 8 --mlp_ratios 1 --layers 1 --time_layers 2 --keep_ratio 0.15 --alpha $AL --tau 2.0 --no-bias \
  --patch_size 1 --token_step 1 --head_pool last --ssa_pe 0 \
  --scheduler cosine -bs $BS --eval_bs 256 -lr $LR --weight_decay 1e-4 --max_lr 0.005 -e $EP $CF --tag $COND lr$LR seed$S bs$BS > "$OUT" 2>&1)
RC=$?
D=$(ls -td "$LOG/SLEEPEDF"/*/*"+${COND}+lr${LR}+seed${S}+bs${BS}+"*"+epoch+${EP}+lr+${LR}+"*/seed${S}_*_f${FOLD}_ctx16* 2>/dev/null | head -1)
CSV="$D/log/final+result.csv"; LCSV="$D/log/last/final+result.csv"
if [ $RC -ne 0 ] || [ ! -s "$CSV" ]; then echo "SKIP(rc=$RC, no result) cond=$COND fold=$FOLD seed=$S lr=$LR out=$OUT" | tee -a "$RES/raw.txt"; exit 1; fi
VALS=$($PY - "$CSV" "$LCSV" <<'PYEOF'
import csv, sys
b = next(csv.DictReader(open(sys.argv[1]))); l = next(csv.DictReader(open(sys.argv[2])))
print(' '.join(f'{k}={v}' for k, v in b.items() if k != 'weights') + ' ' + ' '.join(f'last_{k}={l[k]}' for k in ('score', 'acc', 'macro_f1')))
PYEOF
)
BE=$(tr ' ' ':' < "$D/log/best_epoch.txt" 2>/dev/null)
PAR=$(grep -a 'number of parameters' "$OUT" | grep -oE 'parameters : [0-9]+' | grep -oE '[0-9]+'); t1=$(date +%s)
echo "RESULT exp=nsmt_flif_v1_sleep data=SLEEPEDF cond=$COND fold=$FOLD seed=$S lr=$LR bs=$BS epochs=$EP $VALS best_epoch_val=$BE params=$PAR time_s=$((t1-t0)) sha=$SHA dir=$D out=$OUT" | tee -a "$RES/raw.txt"
