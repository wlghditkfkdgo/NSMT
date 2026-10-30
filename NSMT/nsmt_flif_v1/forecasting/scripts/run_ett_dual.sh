#!/usr/bin/env bash
# run_ett_dual.sh (MODI 2026-10-29, dual-path batch): copy of run_ett_mixer.sh with only the dual-path conditions -- B form
#   spkneo_<lif|flif|pop_lif|pop_flif> (+ spkneo_lif_nomem) and [NSMT hippocampus T=4] nsmt4_<...> (+ nsmt4_lif_nomem), MODI 2026-10-30 + nsmt4q_<...> (neocortex query), the same
#   flags as classification/scripts/run_rhythm_addneo.sh (forecast_wrap.Forecaster kind 'dual'); everything else unchanged.
# MODI 2026-10-27 (generalization batch): one ETT forecasting run of a CinC model of nsmt_flif_v1/classification
# (Spikformer-style SSA or the Neo-mixer), trained by classification/train_forecast.py (classification/forecast_wrap.py:
# window normalisation, channel-independent rows, the model body, flat Linear(N*D -> H) head). Outputs go to this task folder.
# Hyperparameters per dataset = the paper's Table 13 values used by run_ett.sh (patch, emb, heads, mlp, M = mixer layers, lr,
# batch, epochs); the early-stopping patience is 10 and the best-validation weights are tested (stage-2 protocol); both are
# the same for every condition. Plain L1 loss (no model here has the auxiliary decoder).
# usage: COND=<cond> run_ett_mixer.sh <DS: ETTh1|ETTh2|ETTm1|ETTm2> <PL> <seed> <gpu>
#   COND: spk_ssa | mix_<lif|flif|pop_lif|pop_flif>_bidir (OR fusion) | mixlinmx_<lif|flif|pop_lif|pop_flif> (charge-difference fusion)
#         | mixall_<lif|flif|pop_lif|pop_flif>_bidir (OR fusion, the same neuron in every layer; MODI 2026-10-29)
#         | mixlb_<flif|pop_lif|pop_flif>_bidir (spikeDE LIF outside the mixer) | spkall_<n> (Spikformer-style, n everywhere)
#   env:  SUITE EPOCHS PATIENCE (10) RES LOG PY THREADS
set -u
DS=${1:?}; PL=${2:?}; S=${3:?}; GPU=${4:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # task root (nsmt_flif_v1/forecasting)
C="$(cd "$W/../classification" && pwd)"               # model code + train_forecast.py
R="$(cd "$W/../.." && pwd)"                            # NSMT root
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
COND=${COND:?}
case $COND in                     # MODI 2026-10-29: dual-path models, B-form settings (--neo_grad 1.0; no aux loss in train_forecast.py)
  spkneo_lif)        CF="--model myModel --neo_neuron lif --neo_where both --hippo_T 4 --hippo_xattn --hippo_xattn_mode cat --neo_grad 1.0" ;;
  spkneo_flif)       CF="--model myModel --neo_neuron flif --neo_where both --neo_alpha 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode cat --neo_grad 1.0" ;;
  spkneo_pop_lif)    CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode cat --neo_grad 1.0" ;;
  spkneo_pop_flif)   CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode cat --neo_grad 1.0" ;;
  spkneo_lif_nomem)  CF="--model myModel --neo_neuron lif --neo_where both --hippo_T 4 --hippo_xattn --hippo_xattn_mode cat --neo_grad 1.0 --neo_memoryless" ;;
  nsmt4_lif)         CF="--model myModel --neo_neuron lif --neo_where both --hippo_T 4 --hippo_xattn --neo_grad 1.0" ;;
  nsmt4_flif)        CF="--model myModel --neo_neuron flif --neo_where both --neo_alpha 0.5 --hippo_T 4 --hippo_xattn --neo_grad 1.0" ;;
  nsmt4_pop_lif)     CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --neo_grad 1.0" ;;
  nsmt4_pop_flif)    CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --neo_grad 1.0" ;;
  nsmt4_lif_nomem)   CF="--model myModel --neo_neuron lif --neo_where both --hippo_T 4 --hippo_xattn --neo_grad 1.0 --neo_memoryless" ;;
  nsmt4q_lif)        CF="--model myModel --neo_neuron lif --neo_where both --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0" ;;   # MODI 2026-10-30
  nsmt4q_flif)       CF="--model myModel --neo_neuron flif --neo_where both --neo_alpha 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0" ;;
  nsmt4q_pop_lif)    CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0" ;;
  nsmt4q_pop_flif)   CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0" ;;
  nsmt4q_lif_nomem)  CF="--model myModel --neo_neuron lif --neo_where both --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_memoryless" ;;
  *) echo "bad cond $COND"; exit 1 ;;
esac
case $DS in                       # patch emb heads mlp M lr bs epochs   (run_ett.sh; keep_ratio / alpha unused: no aux decoder)
  ETTh1) HP="--patch_size 8 -emb 32 -nh 2  --mlp_ratios 1   --time_layers 2 -lr 0.0005 -bs 64"; E=50 ;;
  ETTh2) HP="--patch_size 8 -emb 64 -nh 8  --mlp_ratios 0.5 --time_layers 1 -lr 0.0005 -bs 64"; E=100 ;;
  ETTm1) HP="--patch_size 8 -emb 64 -nh 8  --mlp_ratios 1   --time_layers 2 -lr 0.001  -bs 64"; E=50 ;;
  ETTm2) HP="--patch_size 8 -emb 64 -nh 16 --mlp_ratios 1   --time_layers 2 -lr 0.0005 -bs 64"; E=50 ;;
  *) echo "bad dataset $DS"; exit 1 ;;
esac
E=${EPOCHS:-$E}; P=${PATIENCE:-10}
SUITE=${SUITE:-flif-v1-ett-dual-$(date +%Y%m%d)}
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}
mkdir -p "$RES" "$LOG"; RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
OUT=$LOG/ett_${DS}_p${PL}_${COND}_seed${S}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$C" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train_forecast.py \
  --log_dir "$LOG/$DS/p$PL" --seed $S -s -nd 0 --gating attn --no-bias --scheduler reduce -e $E --warm_up_epoch 0 \
  --layers 1 --tau 2.0 --weight_decay 0.06 --patience $P \
  --data $DS --data_path $DS.csv --root_path "$W/dataset/ETT-small" --seq_len 96 --pred_len $PL --features M --target OT \
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
echo "RESULT exp=nsmt_flif_v1_ett_dual ds=$DS pl=$PL cond=$COND seed=$S $VALS best_epoch_val_epochs=$BE params=$PAR time_s=$((t1-t0)) sha=$SHA dir=$D" | tee -a "$RES/raw.txt"
