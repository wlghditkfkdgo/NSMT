#!/usr/bin/env bash
# MODI 2026-11-23 (hyperparameter range 2, user decision 11-23): env BS = batch size (unset = 512) and WU = warmup epochs of the
#   cosine schedule, a whole number >= 0 (appends '--lr_warmup $WU'; unset = int(0.2 x epochs) = 6 of 30, as before; 0 = none). When
#   set, the folder tag gains _b<BS> / _w<WU>, the run folder _wu<WU> and RESULT bs= / wu=. Unset: command line, folder and RESULT
#   unchanged.
# MODI 2026-11-21 (4-2 coupling placement, user decision 11-20 "(가) + (나)"): + nsmt4q_pop_<lif|flif>_msn_sde_nc = the base with
#   --neo_current 1 ((가): Neocortex output x learned per-channel alpha into the hippocampus embedding neuron's input current,
#   attention kept) and nsmt4q_pop_<lif|flif>_msn_sde_ncna = + --hippo_attn 0 ((나): no hippocampus attention; the Neocortex is the
#   only token-mixing path). Run folder suffixes _nc / _nc_noattn. Everything else unchanged.
# MODI 2026-11-21 (hyperparameter range 1 adopted by its pre-registered rule; docs/MODI_PROJECT_LOG.md 2026-11-21 01:21 KST): the
#   default is now WDX=1 (--wd_exclude_1d 1; folder tag and run folder _wdx). WDX=0 restores the command line of the runs before
#   11-21 exactly (add PE=1 for the one before 11-20). RESULT always carries wdx=. VPE is ETT only (stops).
# MODI 2026-11-20 (hyperparameter range 1): env WDX=1 appends --wd_exclude_1d 1 (no weight decay on the BatchNorm weights /
#   biases); the folder tag gains _wdx, the run folder _wdx, RESULT wdx=. VPE is ETT only (stops). Unset: unchanged.
# MODI 2026-11-20 (aux-target memory check, user decision 11-20): + nsmt4q_pop_<lif|flif>_msn_sde_{nomem,nshuf}_auxf = the _sde_auxf model
#   with --neo_memoryless / --neo_shuffle; env AUXS = nonzero integer s (_auxf conditions only) appends --aux_shift=s (aux target of token i =
#   token i - s), the folder tag gains _s<s>, the run folder ends in _auxs<s> and RESULT gains auxs=. Unset: unchanged.
# MODI 2026-11-20 (user decision 11-20 "Positional encoding을 OFF하는 것을 default로 하자"): the default is now PE=0 -- no
#   relative position bias ('--ssa_pe 0', run folder suffix _nope). PE=1 restores the bias: the command line of the runs before
#   11-20 exactly. RESULT always carries pe=0 / pe=1. Any other value stops before python.
# MODI 2026-11-19 (positional encoding off, user request 11-19): env PE=0 appends '--ssa_pe 0' (no relative position bias in the
#   attention; the run folder gets the suffix _nope) and RESULT gains pe=0. Unset: command line and RESULT unchanged. Only 0 is accepted.
# MODI 2026-11-19 (full-signal aux, user request 11-19): + nsmt4q_pop_<lif|flif>_{msn,msn_sde}_auxf = the _aux conditions with
#   --keep_ratio 0 (raw patches as the target; it comes after the fixed --keep_ratio 0.15, argparse keeps the last).
# MODI 2026-11-19 (aux loss, user request 11-19): + nsmt4q_pop_<lif|flif>_{msn,msn_sde}_aux = the same flags with AUX = 0.1
#   (loss 0.9 CE + 0.1 reconstruction MSE, keep_ratio 0.15: the ECG framework / NSMT classification default).
# MODI 2026-11-18 (SpikeDE-basis comparison, user decision 11-18 "(가)"): + nsmt4q_pop_<lif|flif>_msn_sde = the msn base with
#   --neuron_all lif (every spikingjelly LIF of the hippocampus -> SpikeDE LIF; the Neocortex keeps its SpikeDE population neuron).
# MODI 2026-11-13 (STFT test): + nsmt4q_pop_<lif|flif>_msn_{stft,fire_stft,fire_same} = msn with the STFT bands as member inputs:
# stft = (na) soma population, tau 4-32; fire_stft = (ga) no soma, 4 independent members with tau 2 (OR); fire_same = (ga) with the unsplit signal.
# MODI 2026-11-13 (frequency bands): + nsmt4q_pop_<lif|flif>_msn_{freq,same,rev} = the Neocortex-MS query population (msn) with
# --neo_band freq / same / rev (one input per member of the population Neocortex embedding); everything else unchanged.
# nsmt_flif_v1: one MIT-BIH run of the ECG model (nsmt_base/classification/ECG) with the neocortex-path neuron replaced.
# MODI 2026-11-08 (membrane shortcut): + nsmt4q_pop_<lif|flif>_{msh,msn,mshn} = the query population with --res_mode ms (hippocampus),
# --neo_res ms (Neocortex) or both; everything else unchanged.
# MODI 2026-11-04 (head test H2): + nsmt4q_pop_<lif|flif>_hattn = the query population with attention pooling in front of the head
# (--head_pool attn); everything else unchanged.
# run_mitbih4.sh (MODI 2026-11-01, query batch on MIT-BIH): copy of run_mitbih3.sh + the query form nsmt4q_<pop_lif|pop_flif>
# (Neocortex as the attention query, hippocampus keys / values; same flags as scripts/run_rhythm_addneo.sh, AUX = 0) and two env
# options: LR (peak learning rate, default 0.01 = unchanged) and VAL (1 = --mitbih_val: fold 0 of a stratified 5-fold split of the
# training file held out as validation, 80 / 20; the last-epoch validation macro-F1 goes into RESULT as val_f1; the test file is still
# tested once with the last-epoch weights). The run folder tag carries COND, LR and VAL so concurrent runs never share a folder.
# run_mitbih3.sh (MODI 2026-10-29, dual-path batch): copy of run_mitbih2.sh + the dual-path models with the B-form settings
# (aux loss weight AUX = 0, --neo_grad 1.0): B form spkneo_<lif|flif|pop_lif|pop_flif> (+ spkneo_lif_nomem) and [NSMT hippocampus
# T=4] nsmt4_<...> (+ nsmt4_lif_nomem), same flags as scripts/run_rhythm_addneo.sh; the other conditions keep AUX = 0.1.
# run_mitbih2.sh (MODI 2026-10-27, generalization batch): copy of run_mitbih.sh + the CinC conditions spk_ssa (Spikformer-style
# SSA, T_h 4), mix_<lif|flif|pop_lif|pop_flif>_bidir (OR) and mixlinmx_<lif|flif|pop_lif|pop_flif> (charge-difference fusion),
# same flags as scripts/run_rhythm6.sh; everything else (ECG framework training, last-epoch test) unchanged.
# Training = the ECG framework's clean 3-seed setting (ECG/scripts/_run_mitbih_3seed.sh): 30 epochs, cosine schedule,
# bs 512, lr 0.01, emb 384, patch 32 (11 tokens), keep_ratio 0.15, aux weight 0.1, CAL-weighted CE; the last-epoch
# weights are tested on the 30% test file (per-epoch "val" columns are that test file; nothing is selected on them).
# Only the condition flags differ.
# usage: COND=<cond> run_mitbih.sh <seed> <gpu>
#   COND: base | neo_lif_both | neo_flif_both | neo_lif_mlp | neo_flif_mlp | no_neo | hippo_only   (default base)
#         stage 6: neo_pop_both | bidir_lif_both | bidir_flif_both | bidir_pop_both  (population f-LIF, bidirectional neocortex)
#   env:  ALPHA (0.5) NEO_TAU (2.0) FDE (pred) H (1.0) MEM (0) POP_TAUS (4,8,16,32) SUITE EPOCHS (30) RES LOG DATA PY THREADS
set -u
S=${1:?}; GPU=${2:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"  # task root (nsmt_flif_v1/classification)
R="$(cd "$W/../.." && pwd)"  # NSMT root
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}   # Python 3.10, torch 1.12.0+cu113, spikingjelly cupy backend
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
COND=${COND:-base}; ALPHA=${ALPHA:-0.5}; AUX=0.1
NEOX="--neo_tau ${NEO_TAU:-2.0} --fde_method ${FDE:-pred} --fde_step ${H:-1.0} --fde_memory ${MEM:-0}"
case $COND in
  base)          CF="" ;;
  neo_lif_both)  CF="--neo_neuron lif  --neo_where both $NEOX" ;;
  neo_flif_both) CF="--neo_neuron flif --neo_where both --neo_alpha $ALPHA $NEOX" ;;
  neo_lif_mlp)   CF="--neo_neuron lif  --neo_where mlp  $NEOX" ;;
  neo_flif_mlp)  CF="--neo_neuron flif --neo_where mlp  --neo_alpha $ALPHA $NEOX" ;;
  no_neo)        CF="--no_neo" ;;                                  # ECG framework ablation: no neocortex, no cross-attention
  hippo_only)    CF="--hippo_only" ;;                              # MODI decision 7: no neocortex, key/value = hippocampus
  neo_pop_both)  CF="--neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_taus ${POP_TAUS:-4,8,16,32} $NEOX" ;;   # stage 6
  bidir_lif_both)  CF="--neo_bidir --neo_neuron lif  --neo_where both $NEOX" ;;
  bidir_flif_both) CF="--neo_bidir --neo_neuron flif --neo_where both --neo_alpha $ALPHA $NEOX" ;;
  bidir_pop_both)  CF="--neo_bidir --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_taus ${POP_TAUS:-4,8,16,32} $NEOX" ;;
  spk_ssa)           CF="--model myModel --hippo_ssa --hippo_T 4" ;;                                # MODI 2026-10-27
  mix_pop_lif_bidir) CF="--model neoMixer --mixer pop_lif --mixer_bidir --pop_soma_init 0.5" ;;
  mix_pop_flif_bidir) CF="--model neoMixer --mixer pop_flif --mixer_bidir --neo_alpha $ALPHA --pop_soma_init 1.0" ;;
  mixlinmx_lif)      CF="--model neoMixer --mixer lif --mixer_bidir --mixer_comb linmx" ;;
  mixlinmx_flif)     CF="--model neoMixer --mixer flif --mixer_bidir --neo_alpha $ALPHA --mixer_comb linmx" ;;
  mixlinmx_pop_lif)  CF="--model neoMixer --mixer pop_lif --mixer_bidir --pop_soma_init 0.5 --mixer_comb linmx" ;;
  mixlinmx_pop_flif) CF="--model neoMixer --mixer pop_flif --mixer_bidir --neo_alpha $ALPHA --pop_soma_init 1.0 --mixer_comb linmx" ;;
  mixall_lif_bidir)  CF="--model neoMixer --mixer lif --mixer_bidir --neo_alpha $ALPHA $NEOX --neuron_scope all" ;;   # MODI 2026-10-29:
  mixall_flif_bidir) CF="--model neoMixer --mixer flif --mixer_bidir --neo_alpha $ALPHA $NEOX --neuron_scope all" ;;  # the neuron in
  mixall_pop_lif_bidir)  CF="--model neoMixer --mixer pop_lif --mixer_bidir --pop_soma_init 0.5 $NEOX --neuron_scope all" ;;   # every layer
  mixall_pop_flif_bidir) CF="--model neoMixer --mixer pop_flif --mixer_bidir --neo_alpha $ALPHA --pop_soma_init 1.0 $NEOX --neuron_scope all" ;;
  mixlb_flif_bidir)  CF="--model neoMixer --mixer flif --mixer_bidir --neo_alpha $ALPHA $NEOX --neuron_scope lifbase" ;;   # control: spikeDE
  mixlb_pop_lif_bidir)  CF="--model neoMixer --mixer pop_lif --mixer_bidir --pop_soma_init 0.5 $NEOX --neuron_scope lifbase" ;;   # LIF outside
  mixlb_pop_flif_bidir) CF="--model neoMixer --mixer pop_flif --mixer_bidir --neo_alpha $ALPHA --pop_soma_init 1.0 $NEOX --neuron_scope lifbase" ;;
  spkneo_lif)        AUX=0; CF="--model myModel --neo_neuron lif --neo_where both --hippo_T 4 --hippo_xattn --hippo_xattn_mode cat --neo_grad 1.0" ;;
  spkneo_flif)       AUX=0; CF="--model myModel --neo_neuron flif --neo_where both --neo_alpha $ALPHA --hippo_T 4 --hippo_xattn --hippo_xattn_mode cat --neo_grad 1.0" ;;
  spkneo_pop_lif)    AUX=0; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode cat --neo_grad 1.0" ;;
  spkneo_pop_flif)   AUX=0; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode cat --neo_grad 1.0" ;;
  spkneo_lif_nomem)  AUX=0; CF="--model myModel --neo_neuron lif --neo_where both --hippo_T 4 --hippo_xattn --hippo_xattn_mode cat --neo_grad 1.0 --neo_memoryless" ;;
  nsmt4_lif)         AUX=0; CF="--model myModel --neo_neuron lif --neo_where both --hippo_T 4 --hippo_xattn --neo_grad 1.0" ;;
  nsmt4_flif)        AUX=0; CF="--model myModel --neo_neuron flif --neo_where both --neo_alpha $ALPHA --hippo_T 4 --hippo_xattn --neo_grad 1.0" ;;
  nsmt4_pop_lif)     AUX=0; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --neo_grad 1.0" ;;
  nsmt4_pop_flif)    AUX=0; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --neo_grad 1.0" ;;
  nsmt4q_pop_lif)    AUX=0; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0" ;;   # MODI 2026-11-01
  nsmt4q_pop_flif)   AUX=0; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0" ;;
  nsmt4q_pop_lif_hattn)  AUX=0; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --head_pool attn" ;;   # MODI 2026-11-04 (H2)
  nsmt4q_pop_flif_hattn) AUX=0; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --head_pool attn" ;;
  nsmt4q_pop_lif_msh)   AUX=0; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --res_mode ms" ;;   # MODI 2026-11-08 (membrane shortcut)
  nsmt4q_pop_lif_msn)   AUX=0; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms" ;;
  nsmt4q_pop_lif_mshn)  AUX=0; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --res_mode ms --neo_res ms" ;;
  nsmt4q_pop_flif_msh)  AUX=0; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --res_mode ms" ;;
  nsmt4q_pop_flif_msn)  AUX=0; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms" ;;
  nsmt4q_pop_lif_msn_sde)   AUX=0; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif" ;;   # MODI 2026-11-18 (SpikeDE basis)
  nsmt4q_pop_flif_msn_sde)  AUX=0; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif" ;;   # MODI 2026-11-18 (SpikeDE basis)
  nsmt4q_pop_lif_msn_sde_nc)     AUX=0; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --neo_current 1" ;;   # MODI 2026-11-21 (4-2 (가))
  nsmt4q_pop_flif_msn_sde_nc)    AUX=0; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --neo_current 1" ;;   # MODI 2026-11-21 (4-2 (가))
  nsmt4q_pop_lif_msn_sde_ncna)   AUX=0; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --neo_current 1 --hippo_attn 0" ;;   # MODI 2026-11-21 (4-2 (나))
  nsmt4q_pop_flif_msn_sde_ncna)  AUX=0; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --neo_current 1 --hippo_attn 0" ;;   # MODI 2026-11-21 (4-2 (나))
  nsmt4q_pop_lif_msn_aux)   AUX=0.1; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms" ;;   # MODI 2026-11-19 (aux loss): NSMT default 0.9 CE + 0.1 MSE
  nsmt4q_pop_lif_msn_sde_aux)   AUX=0.1; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif" ;;   # MODI 2026-11-19 (aux loss): NSMT default 0.9 CE + 0.1 MSE
  nsmt4q_pop_flif_msn_aux)  AUX=0.1; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms" ;;   # MODI 2026-11-19 (aux loss): NSMT default 0.9 CE + 0.1 MSE
  nsmt4q_pop_flif_msn_sde_aux)  AUX=0.1; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif" ;;   # MODI 2026-11-19 (aux loss): NSMT default 0.9 CE + 0.1 MSE
  nsmt4q_pop_lif_msn_auxf)   AUX=0.1; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --keep_ratio 0" ;;   # MODI 2026-11-19 (full-signal aux): AUX 0.1, raw patches (overrides the fixed 0.15)
  nsmt4q_pop_lif_msn_sde_auxf)   AUX=0.1; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --keep_ratio 0" ;;   # MODI 2026-11-19 (full-signal aux): AUX 0.1, raw patches (overrides the fixed 0.15)
  nsmt4q_pop_flif_msn_auxf)  AUX=0.1; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --keep_ratio 0" ;;   # MODI 2026-11-19 (full-signal aux): AUX 0.1, raw patches (overrides the fixed 0.15)
  nsmt4q_pop_flif_msn_sde_auxf)  AUX=0.1; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --keep_ratio 0" ;;   # MODI 2026-11-19 (full-signal aux): AUX 0.1, raw patches (overrides the fixed 0.15)
  nsmt4q_pop_lif_msn_sde_nomem_auxf) AUX=0.1; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --keep_ratio 0 --neo_memoryless" ;;   # MODI 2026-11-20 (aux-target memory check): the _sde_auxf model, Neocortex without memory
  nsmt4q_pop_lif_msn_sde_nshuf_auxf) AUX=0.1; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --keep_ratio 0 --neo_shuffle" ;;   # MODI 2026-11-20 (aux-target memory check): the _sde_auxf model, Neocortex reads the tokens in a shuffled order
  nsmt4q_pop_flif_msn_sde_nomem_auxf) AUX=0.1; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --keep_ratio 0 --neo_memoryless" ;;   # MODI 2026-11-20 (aux-target memory check): the _sde_auxf model, Neocortex without memory
  nsmt4q_pop_flif_msn_sde_nshuf_auxf) AUX=0.1; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --keep_ratio 0 --neo_shuffle" ;;   # MODI 2026-11-20 (aux-target memory check): the _sde_auxf model, Neocortex reads the tokens in a shuffled order
  nsmt4q_pop_flif_mshn) AUX=0; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --res_mode ms --neo_res ms" ;;
  nsmt4q_pop_lif_msn_freq)AUX=0; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band freq" ;;   # MODI 2026-11-13 (frequency bands)
  nsmt4q_pop_lif_msn_same)AUX=0; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band same" ;;
  nsmt4q_pop_lif_msn_rev) AUX=0; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band rev" ;;
  nsmt4q_pop_flif_msn_freq)AUX=0; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band freq" ;;
  nsmt4q_pop_flif_msn_same)AUX=0; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band same" ;;
  nsmt4q_pop_flif_msn_rev)AUX=0; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band rev" ;;
  nsmt4q_pop_lif_msn_stft)     AUX=0; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band stft" ;;   # MODI 2026-11-13 (STFT test): (na) soma, tau 4-32 / (ga) no soma, tau 2 / (ga) control
  nsmt4q_pop_lif_msn_fire_stft)AUX=0; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band stft --neo_pop fire --pop_taus 2,2,2,2" ;;
  nsmt4q_pop_lif_msn_fire_same)AUX=0; CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band same --neo_pop fire --pop_taus 2,2,2,2" ;;
  nsmt4q_pop_flif_msn_stft)    AUX=0; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band stft" ;;
  nsmt4q_pop_flif_msn_fire_stft)AUX=0; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band stft --neo_pop fire --pop_taus 2,2,2,2" ;;
  nsmt4q_pop_flif_msn_fire_same)AUX=0; CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha $ALPHA --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band same --neo_pop fire --pop_taus 2,2,2,2" ;;
  nsmt4_lif_nomem)   AUX=0; CF="--model myModel --neo_neuron lif --neo_where both --hippo_T 4 --hippo_xattn --neo_grad 1.0 --neo_memoryless" ;;
  spkall_lif)        CF="--model myModel --hippo_ssa --hippo_T 4 --neuron_all lif" ;;
  spkall_flif)       CF="--model myModel --hippo_ssa --hippo_T 4 --neuron_all flif --neo_alpha $ALPHA" ;;
  spkall_pop_lif)    CF="--model myModel --hippo_ssa --hippo_T 4 --neuron_all pop_lif --pop_soma_init 0.5" ;;
  spkall_pop_flif)   CF="--model myModel --hippo_ssa --hippo_T 4 --neuron_all pop_flif --neo_alpha $ALPHA --pop_soma_init 1.0" ;;
  mix_*)                                                              # stage 7: mix_<none|lif|flif|pop>_<fwd|bidir>[_L<blocks>]
    IFS=_ read -r _ MK MD ML <<< "$COND"; [ "$MK" = pop ] && MK=pop_flif
    CF="--model neoMixer --mixer $MK --neo_alpha $ALPHA --pop_taus ${POP_TAUS:-4,8,16,32} $NEOX"
    [ "$MD" = bidir ] && CF="$CF --mixer_bidir"
    [ -n "${ML:-}" ] && LAYERS=${ML#L} ;;
  rd_*)                                                               # MODI 2026-11-27 (approved plan 1-snuggly-galaxy.md):
    RB=${COND#rd_}; RX=""; AUX=0                                      # rd_<lif|flif|pop_lif>_<spk|mem|memd>[_acc] = the msn SpikeDE
    [[ $RB == *_acc ]] && { RB=${RB%_acc}; RX="--aux_past rand"; AUX=0.1; }   # base (rd_pop_lif_spk = nsmt4q_pop_lif_msn_sde) with that
    RR=${RB##*_}; RN=${RB%_*}                                         # Neocortex neuron and readout; _acc = accumulation aux, 0.9 CE + 0.1 MSE
    case $RN in lif) NF="--neo_neuron lif --neo_where both" ;; flif) NF="--neo_neuron flif --neo_where both --neo_alpha $ALPHA" ;;
      pop_lif) NF="--neo_neuron pop_lif --neo_where both --pop_soma_init 0.5" ;; *) echo "bad cond $COND"; exit 1 ;; esac
    case $RR in spk) ;; mem|memd) RX="--neo_readout $RR${RX:+ $RX}" ;; *) echo "bad cond $COND"; exit 1 ;; esac
    CF="--model myModel $NF --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif${RX:+ $RX}" ;;
  *) echo "bad cond $COND"; exit 1 ;;
esac
LAYERS=${LAYERS:-1}
PE=${PE:-0}                             # MODI 2026-11-20: default = no relative position bias (user decision 11-20)
if [ "$PE" != 0 ] && [ "$PE" != 1 ]; then echo "PE: 0 (default, no relative position bias) or 1 (bias on, as before 11-20)"; exit 1; fi
PEF=""; [ "$PE" = 0 ] && PEF="--ssa_pe 0"
WDX=${WDX:-1}                           # MODI 2026-11-21: default = hyperparameter range 1 (adopted); WDX=0 = before 11-21
if [ "$WDX" != 0 ] && [ "$WDX" != 1 ]; then echo "WDX: 1 (default, BN / bias without weight decay) or 0 (as before 11-21)"; exit 1; fi
R1F=""; WXT=""; [ "$WDX" = 1 ] && { R1F="--wd_exclude_1d 1"; WXT=_wdx; }
if [ -n "${VPE:-}" ]; then echo "VPE: ETT only"; exit 1; fi
BS=${BS:-}; WU=${WU:-}                  # MODI 2026-11-23 (hyperparameter range 2): unset = unchanged
if [ -n "$BS" ]; then [[ $BS =~ ^[1-9][0-9]*$ ]] || { echo "BS: batch size, a positive integer (unset = 512)"; exit 1; }; fi
if [ -n "$WU" ]; then [[ $WU =~ ^(0|[1-9][0-9]*)$ ]] || { echo "WU: cosine warmup epochs, a whole number >= 0 (unset = 6 of 30)"; exit 1; }; fi
AUXS=${AUXS:-}                          # MODI 2026-11-20 (aux-target memory check): unset = unchanged
if [ -n "$AUXS" ]; then
  [[ $AUXS =~ ^-?[1-9][0-9]*$ ]] || { echo "AUXS: a nonzero integer (unset = the token itself)"; exit 1; }
  [[ $COND == *_auxf ]] || { echo "AUXS: _auxf conditions only"; exit 1; }
fi
SUITE=${SUITE:-flif-v1-mitbih-query-$(date +%Y%m%d)}
EPOCHS=${EPOCHS:-30}
LR=${LR:-0.01}; VAL=${VAL:-0}; VF=""; [ "$VAL" = 1 ] && VF="--mitbih_val"
TG=${COND}_lr${LR}_v${VAL}${AUXS:+_s$AUXS}$WXT${BS:+_b$BS}${WU:+_w$WU}
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}
DATA=${DATA:-$R/nsmt_base/classification/ECG/data}          # 5class_mitbih_{train_7,test_3}_187.csv (local, not in Git)
mkdir -p "$RES" "$LOG"
RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
OUT=$LOG/mitbih_${TG}_seed${S}_${TS}_${SHA}.stdout; t0=$(date +%s)
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train.py \
  --model myModel --seed $S --log_dir "$LOG/MITBIH" --gating attn --attn SSA -s --test --print_epoch 1 -nd 0 \
  -e $EPOCHS --warm_up_epoch 0 --num_channels 1 -bs ${BS:-512} -lr $LR --max_lr 0.005 -emb 384 -nh 8 --mlp_ratios 1 \
  --layers $LAYERS --keep_ratio 0.15 --time_layers 2 --alpha $AUX --patch_size 32 --no-bias --scheduler cosine \
  --data UEA --data_path ECG --seq_len 187 --root_path "$DATA" $CF $VF $PEF $R1F ${WU:+--lr_warmup $WU} ${AUXS:+--aux_shift=$AUXS} --tag $TG seed$S > "$OUT" 2>&1)
RC=$?
D=$(ls -td "$LOG/MITBIH"/*/*"+${TG}+seed${S}+model+"*/seed${S}_*${AUXS:+_auxs$AUXS} 2>/dev/null | head -1)   # the mixer's rho_neo column covers its mixer stacks
CSV="$D/log/final+result.csv"
if [ $RC -ne 0 ] || [ ! -s "$CSV" ]; then echo "SKIP(rc=$RC, no result) cond=$COND seed=$S lr=$LR val=$VAL out=$OUT" | tee -a "$RES/raw.txt"; exit 1; fi
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
VV=""
if [ "$VAL" = 1 ]; then   # last-epoch validation metrics (selection only)
  VV=$($PY - "$D/log/best_log_0.csv" "$EPOCHS" <<'EOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
last = rows[-1]
assert int(last["epoch"]) == int(sys.argv[2]) - 1, "last logged epoch != EPOCHS - 1"
print(f"val_f1={float(last['val_f1']):.6f} val_tot_acc={float(last['val_tot_acc']):.6f}")
EOF
)
  [ -z "$VV" ] && { echo "SKIP(validation parse failed) cond=$COND seed=$S out=$OUT" | tee -a "$RES/raw.txt"; exit 1; }
fi
LRS=$(grep -aoE '^epoch +[0-9]+ \(lr=[0-9.e-]+\)' "$OUT" | grep -oE 'lr=[0-9.e-]+' | cut -d= -f2)
EP=$(echo "$LRS" | grep -c .); LR_FINAL=$(echo "$LRS" | tail -1)
PAR=$(grep -a 'number of parameters' "$OUT" | grep -oE '[0-9]+' | tail -1); t1=$(date +%s)
echo "RESULT exp=nsmt_flif_v1_cls cond=$COND dataset=MITBIH seed=$S lr=$LR val=$VAL $VV $VALS epochs=$EPOCHS epochs_run=$EP lr_final=$LR_FINAL params=$PAR time_s=$((t1-t0)) pe=$PE${AUXS:+ auxs=$AUXS} wdx=$WDX${BS:+ bs=$BS}${WU:+ wu=$WU} sha=$SHA alpha=$ALPHA aux=$AUX dir=$D" | tee -a "$RES/raw.txt"
