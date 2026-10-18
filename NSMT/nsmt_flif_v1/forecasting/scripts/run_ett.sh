#!/usr/bin/env bash
# nsmt_flif_v1: one ETT forecasting run of the NSMT paper model with the neocortex-path neuron replaced.
# Hyperparameters = the paper's Table 13 settings (NSMT/papers/NSMT_code_mapping.md §3), identical to
# nsmt_base/forecasting/scripts/run_ett.sh; only the condition flags differ.
# usage: COND=<cond> run_ett.sh <DS: ETTh1|ETTh2|ETTm1|ETTm2> <PL> <seed> <gpu>
#   COND: base | neo_lif_both | neo_lif_mlp | neo_flif_both | neo_flif_mlp | hippo_only   (default base)
#   env:  ALPHA (0.5) NEO_TAU (2.0) FDE (pred) H (1.0) MEM (0) SUITE EPOCHS RES LOG PY THREADS
#         LEAD (0) NG (0.05): stage-2 regime -- aux target of patch n+LEAD, task-gradient share NG into the neocortex
#         TB (0 = number of patches): backbone spiking steps (stage 3; COND serial_lif | serial_flif | hippo_only)
#         PATIENCE (paper value per dataset): early-stopping patience;  TEST_BEST=1: also test the best-validation checkpoint
set -u
DS=${1:?}; PL=${2:?}; S=${3:?}; GPU=${4:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # task root (nsmt_flif_v1/forecasting)
R="$(cd "$W/../.." && pwd)"  # NSMT root
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}   # Python 3.10, torch 1.12.0+cu113, spikingjelly cupy backend
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
COND=${COND:-base}; ALPHA=${ALPHA:-0.5}
NEOX="--neo_tau ${NEO_TAU:-2.0} --fde_method ${FDE:-pred} --fde_step ${H:-1.0} --fde_memory ${MEM:-0}"
case $COND in
  base)          CF="" ;;
  neo_lif_both)  CF="--neo_neuron lif  --neo_where both $NEOX" ;;
  neo_lif_mlp)   CF="--neo_neuron lif  --neo_where mlp  $NEOX" ;;
  neo_flif_both) CF="--neo_neuron flif --neo_where both --neo_alpha $ALPHA $NEOX" ;;
  neo_flif_mlp)  CF="--neo_neuron flif --neo_where mlp  --neo_alpha $ALPHA $NEOX" ;;
  serial_lif)    CF="--arch serial --neo_neuron lif  --neo_where both $NEOX" ;;               # stage 3
  serial_flif)   CF="--arch serial --neo_neuron flif --neo_where both --neo_alpha $ALPHA $NEOX" ;;
  hippo_only)    CF="--hippo_only" ;;
  linear)        CF="--model linear" ;;                                          # stage 4
  neo_step_*|neo_patch_*)                                                        # stage 4: neo_<unit>_<on|off>_<flat|last>_<lif|flif>
    IFS=_ read -r _ U A RO NN <<< "$COND"
    CF="--model neoOnly --neo_unit $U --align $A --neo_readout $RO --neo_neuron $NN --neo_where both"
    [ "$NN" = flif ] && CF="$CF --neo_alpha $ALPHA"; CF="$CF $NEOX" ;;
  *) echo "bad cond $COND"; exit 1 ;;
esac
LEAD=${LEAD:-0}; NG=${NG:-0.05}; REG=""
if [ "$LEAD" != 0 ] || [ "$NG" != 0.05 ]; then
  [ "$COND" = hippo_only ] && { echo "hippo_only has no neocortex regime"; exit 1; }
  REG="L${LEAD}g${NG}"; CF="$CF --aux_lead $LEAD --neo_grad $NG"
fi
TB=${TB:-0}
if [ "$TB" != 0 ]; then REG="${REG}Tb${TB}"; CF="$CF --backbone_T $TB"; fi
SUITE=${SUITE:-flif-v1-$(date +%Y%m%d)}
EPOCHS=${EPOCHS:-}                                         # empty = the paper value per dataset
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}
mkdir -p "$RES" "$LOG"
RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
case $DS in                       # patch k emb heads alpha mlp M lr bs epoch patience
  ETTh1) HP="--patch_size 8 --keep_ratio 0.25 -emb 32 -nh 2  --alpha 0.8 --mlp_ratios 1   --time_layers 2 -lr 0.0005 -bs 64 --patience 3"; E=50 ;;
  ETTh2) HP="--patch_size 8 --keep_ratio 0.3  -emb 64 -nh 8  --alpha 0.8 --mlp_ratios 0.5 --time_layers 1 -lr 0.0005 -bs 64 --patience 2"; E=100 ;;
  ETTm1) HP="--patch_size 8 --keep_ratio 0.25 -emb 64 -nh 8  --alpha 0.5 --mlp_ratios 1   --time_layers 2 -lr 0.001  -bs 64 --patience 3"; E=50 ;;
  ETTm2) HP="--patch_size 8 --keep_ratio 0.25 -emb 64 -nh 16 --alpha 0.5 --mlp_ratios 1   --time_layers 2 -lr 0.0005 -bs 64 --patience 2"; E=50 ;;
  *) echo "bad dataset $DS"; exit 1 ;;
esac
E=${EPOCHS:-$E}
P=$(echo "$HP" | grep -oE -- '--patience [0-9]+' | awk '{print $2}')
if [ -n "${PATIENCE:-}" ]; then P=$PATIENCE; HP="$HP --patience $P"; fi     # argparse keeps the last --patience
TB=""; [ "${TEST_BEST:-0}" = 1 ] && TB="--test_best"
IX="ett_${DS}_p${PL}_${COND}${REG:+_$REG}"; OUT=$LOG/${IX}_seed${S}_${TS}_${SHA}.stdout; t0=$(date +%s)
# --log_dir is resolved relative to the code folder by config.py; --tag puts cond+seed into the run-dir name.
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train.py \
  --log_dir "$LOG" --model myModel --seed $S --gating attn --no-bias --scheduler reduce -s --test -nd 0 \
  -e $E --warm_up_epoch 0 --layers 1 --max_ratio 2 --tau 2.0 \
  --data $DS --data_path $DS.csv --root_path "$W/dataset/ETT-small" --seq_len 96 --pred_len $PL --c_in 7 \
  --features M --target OT $HP $CF $TB --tag $COND seed$S $REG > "$OUT" 2>&1)
RC=$?
D=$(ls -td "$LOG/$DS"/*/*"+${COND}+seed${S}+${REG:+$REG+}model+"*"+pred_len+${PL}+"* 2>/dev/null | head -1)
CSV="$D/log/final+result.csv"
col() { sed -n 2p "$CSV" 2>/dev/null | cut -d, -f$1; }
MSE=$(col 3); MAE=$(col 4); EN=$(col 10); NLIF=$(col 14); NFFT=$(col 15); NDIR=$(col 16); TFFT=$(col 17); TDIR=$(col 18); RNEO=$(col 19); RHIP=$(col 20)
# a failed training run is reported with its return code before any best-checkpoint lookup
if [ $RC -ne 0 ] || [ -z "$MSE" ]; then echo "SKIP(rc=$RC, no result) cond=$COND ds=$DS pl=$PL seed=$S out=$OUT" | tee -a "$RES/raw.txt"; exit 1; fi
LRS=$(grep -aoE '^epoch +[0-9]+ \(lr=[0-9.e-]+\)' "$OUT" | grep -oE 'lr=[0-9.e-]+' | cut -d= -f2)
LR_FINAL=$(echo "$LRS" | tail -1); LR_DROPS=$(( $(echo "$LRS" | uniq | wc -l) - 1 ))
EP=$(echo "$LRS" | grep -c .)                                    # epochs actually trained (the old count also matched the 'epoch---' argument line)
BEST=""
if [ -n "$TB" ]; then
  BCSV="$D/log/best/final+result.csv"; bcol() { sed -n 2p "$BCSV" 2>/dev/null | cut -d, -f$1; }
  BEST="mse_best=$(bcol 3) mae_best=$(bcol 4) syn_energy_best=$(bcol 10 | tr -d ' uJ') total_fft_uJ_best=$(bcol 17) total_direct_uJ_best=$(bcol 18) rho_neo_best=$(bcol 19) rho_hippo_best=$(bcol 20) best_epoch=$(cat "$D/log/best/best_epoch.txt" 2>/dev/null)"
  [ -z "$(bcol 3)" ] && { echo "SKIP(best-checkpoint test missing) cond=$COND ds=$DS pl=$PL seed=$S out=$OUT" | tee -a "$RES/raw.txt"; exit 1; }
fi
PAR=$(grep -a 'number of parameters' "$OUT" | grep -oE '[0-9]+' | tail -1); t1=$(date +%s)
echo "RESULT exp=nsmt_flif_v1 cond=$COND regime=${REG:-L0g0.05} dataset=${DS}_p${PL} seed=$S mse=$MSE mae=$MAE syn_energy=$EN neuron_lif_uJ=$NLIF neuron_fft_uJ=$NFFT neuron_direct_uJ=$NDIR total_fft_uJ=$TFFT total_direct_uJ=$TDIR rho_neo=$RNEO rho_hippo=$RHIP epochs=$E epochs_run=$EP patience=$P lr_final=$LR_FINAL lr_drops=$LR_DROPS $BEST params=$PAR time_s=$((t1-t0)) sha=$SHA alpha=$ALPHA dir=$D" | tee -a "$RES/raw.txt"
