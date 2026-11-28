#!/usr/bin/env bash
# MODI 2026-11-24 (fix): the run folder lookup after training always carries the lr: +lr+<LR>+, or +lr+<the dataset's -lr
#   above>+ when LR is unset. Before, with LR unset the pattern had no lr, matched the folders of every lr with the same tag
#   (range-2 search: LR=0.00025 / 0.001 runs next to the default-lr run) and took the newest; 5 finished default-lr runs got a
#   SKIP line (docs/MODI_PROJECT_LOG.md 2026-11-24). Command line, folders and RESULT line unchanged.
# MODI 2026-11-23 (hyperparameter range 2, user decision 11-23): env BS = batch size (appends '-bs $BS' after the dataset flags,
#   argparse keeps the last; unset = 64) and WU = lr warmup in epochs, 0 < WU <= 1 (appends '--lr_warmup $WU': linear from lr / W to
#   lr over the first W = round(WU x steps per epoch) optimizer steps; unset = none). When set, the folder tag becomes
#   <COND>_b<BS>_w<WU> (so runs that differ only in BS / WU never share a folder), the run folder gains _wu<WU> and RESULT gains
#   bs= / wu=. Unset: command line, folder and RESULT unchanged.
# MODI 2026-11-23 (fix, same day): the stdout file name now carries BS / WU / LR (ett_..._<COND>_b<BS>_w<WU>_lr<LR>_seed...). Before,
#   two concurrent runs of one model / cell / seed that differed only in these wrote the same stdout file (the first range-2 launch,
#   stopped after 3 minutes). Unset: the name is unchanged.
# MODI 2026-11-21 (4-2 coupling placement, user decision 11-20 "(가) + (나)"): + nsmt4q_pop_<lif|flif>_msn_sde_nc = the base with
#   --neo_current 1 ((가): Neocortex output x learned per-channel alpha into the hippocampus embedding neuron's input current,
#   attention kept) and nsmt4q_pop_<lif|flif>_msn_sde_ncna = + --hippo_attn 0 ((나): no hippocampus attention; the Neocortex is the
#   only token-mixing path). Run folder suffixes _nc / _nc_noattn. Everything else unchanged.
# MODI 2026-11-21 (hyperparameter range 1 adopted by its pre-registered rule; docs/MODI_PROJECT_LOG.md 2026-11-21 01:21 KST): the
#   defaults are now WDX=1 (--wd_exclude_1d 1) and VPE=4 (--val_per_epoch 4). WDX=0 VPE=1 restores the command line of the runs
#   before 11-21 exactly (add PE=1 for the one before 11-20). RESULT always carries wdx= / vpe=. Any other value stops before python.
# MODI 2026-11-20 (hyperparameter range 1): env WDX=1 appends --wd_exclude_1d 1 (no weight decay on the BatchNorm
#   weights / biases), VPE=K (2-9) appends --val_per_epoch K (K validations per epoch for the best checkpoint / early stopping);
#   the run folder gains _wdx / _vpe<K>, RESULT gains wdx= / vpe=. Unset: unchanged.
# MODI 2026-11-20 (aux-target memory check, user decision 11-20): + nsmt4q_pop_<lif|flif>_msn_sde_{nomem,nshuf}_auxf = the _sde_auxf model
#   with --neo_memoryless (every token through the Neocortex alone) / --neo_shuffle (shuffled token order); env AUXS = nonzero integer s (_auxf conditions only) appends
#   --aux_shift=s (aux target of token i = token i - s), the run folder ends in _auxs<s> and RESULT gains auxs=. Unset: unchanged.
# MODI 2026-11-20 (user decision 11-20 "Positional encoding을 OFF하는 것을 default로 하자"): the default is now PE=0 -- no
#   relative position bias ('--ssa_pe 0', run folder suffix _nope). PE=1 restores the bias: the command line of the runs before
#   11-20 exactly. RESULT always carries pe=0 / pe=1. Any other value stops before python.
# MODI 2026-11-19 (positional encoding off, user request 11-19): env PE=0 appends '--ssa_pe 0' (no relative position bias in the
#   attention; the run folder gets the suffix _nope) and RESULT gains pe=0. Unset: command line and RESULT unchanged. Only 0 is accepted.
# MODI 2026-11-19 (full-signal aux, user request 11-19): + nsmt4q_pop_<lif|flif>_{msn,msn_sde}_auxf = the _aux conditions with the
#   raw patches as the reconstruction target (--keep_ratio 0), same alpha.
# MODI 2026-11-19 (overfitting test): env WD = weight decay (--weight_decay, unset = 0.06 as before) and DROP = head dropout
#   (--head_dropout, appended only when set); RESULT gains wd= / drop= only when set. Unset: command line and RESULT unchanged.
# MODI 2026-11-19 (aux loss, user request 11-19): + nsmt4q_pop_<lif|flif>_{msn,msn_sde}_aux = the msn base (spikingjelly hippocampus) /
#   the all-SpikeDE msn base with NSMT's reconstruction loss (--aux_alpha, --keep_ratio as run_ett.sh: ETTh2 0.8 / 0.3, ETTm2 0.5 / 0.25).
# MODI 2026-11-18 (SpikeDE-basis comparison, user decision 11-18 "(가)"): + nsmt4q_pop_<lif|flif>_msn_sde = the msn base with
#   --neuron_all lif (every spikingjelly LIF of the hippocampus -> SpikeDE LIF; the Neocortex keeps its SpikeDE population neuron).
# MODI 2026-11-18 (long-input lr test): env LR = learning rate (unset = the dataset default above, every earlier run; the command
#   line, folder glob and RESULT line are then unchanged). Set: '-lr $LR' is appended after the dataset flags (argparse keeps the
#   last), the run folder is found with '+epoch+E+lr+$LR+' (LR must be written as Python prints the float, e.g. 0.0001, 0.00025,
#   5e-05; otherwise no folder matches and the run is a SKIP), and RESULT gains lr=.
# MODI 2026-11-18 (coupling attention test): + nsmt4q_pop_<lif|flif>_msn_xl = the msn base with --attn_score xnor_log (XNOR + Log-PE).
# MODI 2026-11-16 (input-length test): env SEQ = input length (--seq_len, default 96 = every earlier run; RESULT gains seq_len=);
# + nsmt4q_pop_<lif|flif>_msn_{nomem,nshuf} = the msn base model with --neo_memoryless (every token alone, no state crosses
# tokens) / --neo_shuffle (the Neocortex reads the tokens in a shuffled order, outputs put back): the controls of the T = N memory.
# MODI 2026-11-13 (STFT test): + nsmt4q_pop_<lif|flif>_msn_{stft,fire_stft,fire_same} = msn with the STFT bands as member inputs:
# stft = (na) soma population, tau 4-32; fire_stft = (ga) no soma, 4 independent members with tau 2 (OR); fire_same = (ga) with the unsplit signal.
# MODI 2026-11-13 (frequency bands): + nsmt4q_pop_<lif|flif>_msn_{freq,same,rev} = the Neocortex-MS query population (msn) with
# --neo_band freq (DCT bands, lowest -> slowest member) / same (unsplit signal, 4 separate embeddings) / rev (lowest -> fastest).
# MODI 2026-11-08 (membrane shortcut): + nsmt4q_pop_<lif|flif>_{msh,msn,mshn} = the query population with --res_mode ms (hippocampus),
# --neo_res ms (Neocortex) or both; everything else unchanged.
# MODI 2026-11-04 (head test H6): + nsmt4q_pop_<lif|flif>_h6 = the query population with the H6 head (linear shortcut + small head,
# --fc_head h6 --fc_head_dim 8; final+result.csv adds mse_lin / mae_lin = the shortcut alone); everything else unchanged.
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
  nsmt4q_pop_lif_h6)  CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --fc_head h6 --fc_head_dim 8" ;;   # MODI 2026-11-04 (H6)
  nsmt4q_pop_flif_h6) CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --fc_head h6 --fc_head_dim 8" ;;
  nsmt4q_pop_lif_msh)   CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --res_mode ms" ;;   # MODI 2026-11-08: membrane
  nsmt4q_pop_lif_msn)   CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms" ;;    # shortcut in the hippocampus
  nsmt4q_pop_lif_mshn)  CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --res_mode ms --neo_res ms" ;;   # (h), Neocortex (n), both
  nsmt4q_pop_flif_msh)  CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --res_mode ms" ;;
  nsmt4q_pop_flif_msn)  CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms" ;;
  nsmt4q_pop_flif_mshn) CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --res_mode ms --neo_res ms" ;;
  nsmt4q_pop_lif_msn_nomem)CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_memoryless" ;;   # MODI 2026-11-16 (input-length test): memory / order controls of msn
  nsmt4q_pop_lif_msn_nshuf)CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_shuffle" ;;
  nsmt4q_pop_flif_msn_nomem)CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_memoryless" ;;
  nsmt4q_pop_flif_msn_nshuf)CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_shuffle" ;;
  nsmt4q_pop_lif_msn_xl) CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --attn_score xnor_log" ;;   # MODI 2026-11-18 (coupling attention test): msn + XNOR + Log-PE score
  nsmt4q_pop_flif_msn_xl)CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --attn_score xnor_log" ;;
  nsmt4q_pop_lif_msn_sde)   CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif" ;;   # MODI 2026-11-18 (SpikeDE basis)
  nsmt4q_pop_flif_msn_sde)  CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif" ;;   # MODI 2026-11-18 (SpikeDE basis)
  nsmt4q_pop_lif_msn_sde_nc)     CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --neo_current 1" ;;   # MODI 2026-11-21 (4-2 (가))
  nsmt4q_pop_flif_msn_sde_nc)    CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --neo_current 1" ;;   # MODI 2026-11-21 (4-2 (가))
  nsmt4q_pop_lif_msn_sde_ncna)   CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --neo_current 1 --hippo_attn 0" ;;   # MODI 2026-11-21 (4-2 (나))
  nsmt4q_pop_flif_msn_sde_ncna)  CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --neo_current 1 --hippo_attn 0" ;;   # MODI 2026-11-21 (4-2 (나))
  nsmt4q_pop_lif_msn_aux)   CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms" ;;   # MODI 2026-11-19 (aux loss): + NSMT reconstruction loss (per dataset below)
  nsmt4q_pop_lif_msn_sde_aux)   CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif" ;;   # MODI 2026-11-19 (aux loss): + NSMT reconstruction loss (per dataset below)
  nsmt4q_pop_flif_msn_aux)  CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms" ;;   # MODI 2026-11-19 (aux loss): + NSMT reconstruction loss (per dataset below)
  nsmt4q_pop_flif_msn_sde_aux)  CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif" ;;   # MODI 2026-11-19 (aux loss): + NSMT reconstruction loss (per dataset below)
  nsmt4q_pop_lif_msn_auxf)   CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms" ;;   # MODI 2026-11-19 (full-signal aux): keep_ratio 0 (below)
  nsmt4q_pop_lif_msn_sde_auxf)   CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif" ;;   # MODI 2026-11-19 (full-signal aux): keep_ratio 0 (below)
  nsmt4q_pop_flif_msn_auxf)  CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms" ;;   # MODI 2026-11-19 (full-signal aux): keep_ratio 0 (below)
  nsmt4q_pop_flif_msn_sde_auxf)  CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif" ;;   # MODI 2026-11-19 (full-signal aux): keep_ratio 0 (below)
  nsmt4q_pop_lif_msn_sde_nomem_auxf) CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --neo_memoryless" ;;   # MODI 2026-11-20 (aux-target memory check): the _sde_auxf model, Neocortex without memory
  nsmt4q_pop_lif_msn_sde_nshuf_auxf) CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --neo_shuffle" ;;   # MODI 2026-11-20 (aux-target memory check): the _sde_auxf model, Neocortex reads the tokens in a shuffled order
  nsmt4q_pop_flif_msn_sde_nomem_auxf) CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --neo_memoryless" ;;   # MODI 2026-11-20 (aux-target memory check): the _sde_auxf model, Neocortex without memory
  nsmt4q_pop_flif_msn_sde_nshuf_auxf) CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif --neo_shuffle" ;;   # MODI 2026-11-20 (aux-target memory check): the _sde_auxf model, Neocortex reads the tokens in a shuffled order
  nsmt4q_pop_lif_msn_freq)CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band freq" ;;   # MODI 2026-11-13 (frequency bands): msn + one input per population member
  nsmt4q_pop_lif_msn_same)CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band same" ;;
  nsmt4q_pop_lif_msn_rev) CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band rev" ;;
  nsmt4q_pop_flif_msn_freq)CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band freq" ;;
  nsmt4q_pop_flif_msn_same)CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band same" ;;
  nsmt4q_pop_flif_msn_rev)CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band rev" ;;
  nsmt4q_pop_lif_msn_stft)     CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band stft" ;;   # MODI 2026-11-13 (STFT test): (na) soma, tau 4-32 / (ga) no soma, tau 2 / (ga) control
  nsmt4q_pop_lif_msn_fire_stft)CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band stft --neo_pop fire --pop_taus 2,2,2,2" ;;
  nsmt4q_pop_lif_msn_fire_same)CF="--model myModel --neo_neuron pop_lif --neo_where both --pop_soma_init 0.5 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band same --neo_pop fire --pop_taus 2,2,2,2" ;;
  nsmt4q_pop_flif_msn_stft)    CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band stft" ;;
  nsmt4q_pop_flif_msn_fire_stft)CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band stft --neo_pop fire --pop_taus 2,2,2,2" ;;
  nsmt4q_pop_flif_msn_fire_same)CF="--model myModel --neo_neuron pop_flif --neo_where both --neo_alpha 0.5 --pop_soma_init 1.0 --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neo_band same --neo_pop fire --pop_taus 2,2,2,2" ;;
  rd_*)                           # MODI 2026-11-27 (approved plan 1-snuggly-galaxy.md): rd_<lif|flif|pop_lif>_<spk|mem|memd>[_acc] = the
    RB=${COND#rd_}; RX=""           # msn SpikeDE base (rd_pop_lif_spk = nsmt4q_pop_lif_msn_sde) with that Neocortex neuron, Neocortex readout
    [[ $RB == *_acc ]] && { RB=${RB%_acc}; RX="--aux_past rand --past_aux_w 0.1"; }   # (spikes / membrane via q_linear / membrane as the
    RR=${RB##*_}; RN=${RB%_*}                                                          # queries) and, with _acc, the accumulation aux
    case $RN in lif) NF="--neo_neuron lif --neo_where both" ;; flif) NF="--neo_neuron flif --neo_where both --neo_alpha 0.5" ;;
      pop_lif) NF="--neo_neuron pop_lif --neo_where both --pop_soma_init 0.5" ;; *) echo "bad cond $COND"; exit 1 ;; esac
    case $RR in spk) ;; mem|memd) RX="--neo_readout $RR${RX:+ $RX}" ;; *) echo "bad cond $COND"; exit 1 ;; esac
    CF="--model myModel $NF --hippo_T 4 --hippo_xattn --hippo_xattn_mode query --neo_grad 1.0 --neo_res ms --neuron_all lif${RX:+ $RX}" ;;
  *) echo "bad cond $COND"; exit 1 ;;
esac
case $DS in                       # patch emb heads mlp M lr bs epochs   (run_ett.sh; keep_ratio / alpha unused: no aux decoder)
  ETTh1) HP="--patch_size 8 -emb 32 -nh 2  --mlp_ratios 1   --time_layers 2 -lr 0.0005 -bs 64"; E=50 ;;
  ETTh2) HP="--patch_size 8 -emb 64 -nh 8  --mlp_ratios 0.5 --time_layers 1 -lr 0.0005 -bs 64"; E=100 ;;
  ETTm1) HP="--patch_size 8 -emb 64 -nh 8  --mlp_ratios 1   --time_layers 2 -lr 0.001  -bs 64"; E=50 ;;
  ETTm2) HP="--patch_size 8 -emb 64 -nh 16 --mlp_ratios 1   --time_layers 2 -lr 0.0005 -bs 64"; E=50 ;;
  *) echo "bad dataset $DS"; exit 1 ;;
esac
if [[ $COND == *_auxf ]]; then          # MODI 2026-11-19: the same alpha, target = the raw patches (keep_ratio 0)
  case $DS in ETTh2) CF="$CF --keep_ratio 0 --aux_alpha 0.8" ;; ETTm2) CF="$CF --keep_ratio 0 --aux_alpha 0.5" ;;
    *) echo "auxf: no NSMT setting for $DS"; exit 1 ;; esac
fi
if [[ $COND == *_aux ]]; then           # MODI 2026-11-19: NSMT forecasting aux setting (run_ett.sh / paper Table 13): keep_ratio, alpha
  case $DS in ETTh2) CF="$CF --keep_ratio 0.3 --aux_alpha 0.8" ;; ETTm2) CF="$CF --keep_ratio 0.25 --aux_alpha 0.5" ;;
    *) echo "aux: no NSMT setting for $DS"; exit 1 ;; esac
fi
E=${EPOCHS:-$E}; P=${PATIENCE:-10}; SEQ=${SEQ:-96}
PE=${PE:-0}                             # MODI 2026-11-20: default = no relative position bias (user decision 11-20)
if [ "$PE" != 0 ] && [ "$PE" != 1 ]; then echo "PE: 0 (default, no relative position bias) or 1 (bias on, as before 11-20)"; exit 1; fi
PEF=""; [ "$PE" = 0 ] && PEF="--ssa_pe 0"
WDX=${WDX:-1}; VPE=${VPE:-4}            # MODI 2026-11-21: default = hyperparameter range 1 (adopted); WDX=0 VPE=1 = before 11-21
if [ "$WDX" != 0 ] && [ "$WDX" != 1 ]; then echo "WDX: 1 (default, BN / bias without weight decay) or 0 (as before 11-21)"; exit 1; fi
[[ $VPE =~ ^[1-9]$ ]] || { echo "VPE: validations per epoch, 4 (default), 1 (as before 11-21) or 2-9"; exit 1; }
R1F=""; [ "$WDX" = 1 ] && R1F="--wd_exclude_1d 1"; [ "$VPE" != 1 ] && R1F="$R1F --val_per_epoch $VPE"
BS=${BS:-}; WU=${WU:-}                  # MODI 2026-11-23 (hyperparameter range 2): unset = unchanged
if [ -n "$BS" ]; then [[ $BS =~ ^[1-9][0-9]*$ ]] || { echo "BS: batch size, a positive integer (unset = 64)"; exit 1; }; fi
if [ -n "$WU" ]; then [[ $WU =~ ^(1|0\.[0-9]*[1-9][0-9]*)$ ]] || { echo "WU: lr warmup in epochs, 0 < WU <= 1 (unset = none)"; exit 1; }; fi
TX="${BS:+_b$BS}${WU:+_w$WU}"; R2F="${BS:+-bs $BS}${WU:+ --lr_warmup $WU}"
LRF=${LR:-$(sed -nE 's/.* -lr ([^ ]+) .*/\1/p' <<<"$HP")}   # MODI 2026-11-24: the lr of the folder lookup (LR, else the dataset's -lr in HP)
[ -n "$LRF" ] || { echo "no lr for the run folder lookup"; exit 1; }
AUXS=${AUXS:-}                          # MODI 2026-11-20 (aux-target memory check): unset = unchanged
if [ -n "$AUXS" ]; then
  [[ $AUXS =~ ^-?[1-9][0-9]*$ ]] || { echo "AUXS: a nonzero integer (unset = the token itself)"; exit 1; }
  [[ $COND == *_auxf ]] || { echo "AUXS: _auxf conditions only"; exit 1; }
fi
SUITE=${SUITE:-flif-v1-ett-dual-$(date +%Y%m%d)}
RES=${RES:-$W/results/$SUITE}; LOG=${LOG:-$W/log/$SUITE}
mkdir -p "$RES" "$LOG"; RES="$(cd "$RES" && pwd)"; LOG="$(cd "$LOG" && pwd)"
SHA=$(cd "$R" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
OUT=$LOG/ett_${DS}_p${PL}_${COND}${TX}${LR:+_lr$LR}_seed${S}_${TS}_${SHA}.stdout; t0=$(date +%s)   # MODI 2026-11-23: + BS / WU / LR (runs that differ only in them wrote the same stdout file)
(cd "$C" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY ./train_forecast.py \
  --log_dir "$LOG/$DS/p$PL" --seed $S -s -nd 0 --gating attn --no-bias --scheduler reduce -e $E --warm_up_epoch 0 \
  --layers 1 --tau 2.0 --weight_decay ${WD:-0.06} --patience $P \
  --data $DS --data_path $DS.csv --root_path "$W/dataset/ETT-small" --seq_len $SEQ --pred_len $PL --features M --target OT \
  $HP $CF ${LR:+-lr $LR} ${DROP:+--head_dropout $DROP} $PEF $R1F $R2F ${AUXS:+--aux_shift=$AUXS} --tag $COND$TX seed$S > "$OUT" 2>&1)
RC=$?
D=$(ls -td "$LOG/$DS/p$PL"/*/*"+${COND}${TX}+seed${S}+"*"+epoch+${E}+lr+${LRF}+"*/seed${S}_*${AUXS:+_auxs$AUXS} 2>/dev/null | head -1)
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
echo "RESULT exp=nsmt_flif_v1_ett_dual ds=$DS pl=$PL cond=$COND seed=$S $VALS best_epoch_val_epochs=$BE params=$PAR time_s=$((t1-t0)) seq_len=$SEQ${LR:+ lr=$LR}${WD:+ wd=$WD}${DROP:+ drop=$DROP} wdx=$WDX vpe=$VPE${BS:+ bs=$BS}${WU:+ wu=$WU}${AUXS:+ auxs=$AUXS} pe=$PE sha=$SHA dir=$D" | tee -a "$RES/raw.txt"
