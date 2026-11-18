#!/usr/bin/env bash
# make_queue_pn.sh (2026-11-18, exp/f-lif-pop-v3; PROJECT_LOG neuron implementation / placement pre-registration; user request 11-18).
#   SF-de = the Spikformer-style model with the SpkMixer's spikeDE LIF in every layer (NALL=lif, tag _nalif; user choice 11-18), tuned on
#           the Spikformer-style grid of 11-01: ETT lr {1e-4, 2.5e-4, 5e-4*, 1e-3, 2e-3} (layers ETTh2 1, ETTm2 2 as Table 13), MIT-BIH lr
#           {0.0025, 0.005, 0.01*, 0.02} on validation fold 0 (2 layers), then retrained on the whole training file at the selected lr.
#   grid  = SpkMixer neuron x placement x repeat pooling, the 9 combinations not run before (tuned LIF SpkMixer settings, not re-tuned:
#           ETT muxlin + IAND, 1 mixer layer per direction, lr 5e-4 ETTh2 / 1e-4 ETTm2; MIT-BIH muxlin + ADD, 2 layers, lr 0.02):
#             f-LIF (all), f-LIF + mean, pop LIF + mean, pop LIF mixer-only, pop LIF mixer-only + mean,
#             pop f-LIF (all), pop f-LIF + mean, pop f-LIF mixer-only, pop f-LIF mixer-only + mean
#           (mixer-only = NSCOPE=mixer: population in the directional stacks + fusion, user choice 11-18; the rest the single neuron)
#   len336 = ETT input 336: SF-de at its selected lr and pop LIF mixer-only (re-run candidates of the 11-17 input-length test)
# Stages:
#   repro    the default paths again, seed 42: LIF / pop LIF / Spikformer-style (spikingjelly) ETTh2 H96, LIF and Spikformer-style MIT-BIH
#   smoke    2 epochs ETT (ETTh2 H96, ETTm2 H720) and 1 epoch MIT-BIH, seed 42: every grid combination, SF-de (MIT-BIH also on fold 0),
#            input 336 SF-de and pop LIF mixer-only on ETTh2 H96 = 33 runs
#   sf       SF-de tuning runs: ETT 5 lr x 4 cells x 3 seeds = 60 (suite flif-v1-ett-sfde), MIT-BIH 4 lr x 3 seeds on fold 0 = 12 (flif-v1-mitbih-sfde)
#   main     needs the selections (summarize_pn.py --stage select): SF-de MIT-BIH final 3, then the grid ETT 9 x 12 = 108
#            (flif-v1-ett-spkmux-pn), MIT-BIH 9 x 3 = 27 (flif-v1-mitbih-spkmux-pn)
#   len336   needs the ETT selection: SF-de 12 + pop LIF mixer-only 12 = 24 (flif-v1-ett-spkmux-pn336)
#   sf336    (user decision 11-18 21:4x: grid stopped, pop LIF mixer-only 336 dropped) SF-de 336 only = 12 (flif-v1-ett-spkmux-pn336)
# usage (from anywhere): bash make_queue_pn.sh repro|smoke|sf|main|len336|sf336 ; prints <queue file>:<jobs>
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q="$W/scripts/queues"; mkdir -p "$Q"
E="bash cinc_mixer_v1/scripts/run_ett_spk.sh"
M="bash cinc_mixer_v1/scripts/run_mitbih_spk.sh"
D=20261118
lr_mx() { case $1 in ETTh2) echo 0.0005 ;; ETTm2) echo 0.0001 ;; esac; }
ly_sf() { case $1 in ETTh2) echo 1 ;; ETTm2) echo 2 ;; esac; }
GRID=("COND=spkmx_pop_flif_muxlin_X" "COND=spkmx_pop_flif_muxlin_X MIXPOOL=mean" "COND=spkmx_pop_flif_muxlin_X NSCOPE=mixer"
      "COND=spkmx_pop_flif_muxlin_X NSCOPE=mixer MIXPOOL=mean" "COND=spkmx_pop_lif_muxlin_X MIXPOOL=mean" "COND=spkmx_pop_lif_muxlin_X NSCOPE=mixer"
      "COND=spkmx_pop_lif_muxlin_X NSCOPE=mixer MIXPOOL=mean" "COND=spkmx_flif_muxlin_X" "COND=spkmx_flif_muxlin_X MIXPOOL=mean")
g_ett() { echo "$1 ${2//_X/_iand} LR=$(lr_mx $3) LAYERS=1 $E $3 $4 $5"; }          # <prefix> <combination> <ds> <pl> <seed>
g_mit() { echo "$1 ${2//_X/_add} LR=0.02 MLAYERS=2 $M $3"; }                        # <prefix> <combination> <seed>
sf_ett() { echo "$1 COND=spk_ssa NALL=lif LR=$2 LAYERS=$(ly_sf $3) $E $3 $4 $5"; }   # <prefix> <lr> <ds> <pl> <seed>
sf_mit() { echo "$1 COND=spk_ssa NALL=lif LR=$2 MLAYERS=2 $M $3"; }                 # <prefix> <lr> <seed>
sel() {                                                                             # sel ett <ds> | sel mitbih MITBIH -> selected lr (fail closed)
  local f="$W/results/flif-v1-$1-sfde/selected_lr.json"
  [ -s "$f" ] || { echo "no selection $f" >&2; exit 1; }
  python3 -c "import json; print(json.load(open('$f'))['selected']['$2'])" || exit 1
}
case ${1:?repro|smoke|sf|main|len336|sf336} in
  repro) f="$Q/pn-repro-$D.txt"; : > "$f"
         echo "SUITE=check-pn-repro-$D-ett COND=spkmx_lif_muxlin_iand LR=0.0005 LAYERS=1 $E ETTh2 96 42" >> "$f"
         echo "SUITE=check-pn-repro-$D-ett COND=spkmx_pop_lif_muxlin_iand LR=0.0005 LAYERS=1 $E ETTh2 96 42" >> "$f"
         echo "SUITE=check-pn-repro-$D-ett COND=spk_ssa LR=0.0005 LAYERS=1 $E ETTh2 96 42" >> "$f"
         echo "SUITE=check-pn-repro-$D COND=spkmx_lif_muxlin_add LR=0.02 MLAYERS=2 $M 42" >> "$f"
         echo "SUITE=check-pn-repro-$D COND=spk_ssa LR=0.005 MLAYERS=2 $M 42" >> "$f" ;;
  smoke) f="$Q/pn-smoke-$D.txt"; : > "$f"
         for c in "${GRID[@]}"; do for dp in "ETTh2 96" "ETTm2 720"; do g_ett "EPOCHS=2 SUITE=smoke-pn-ett-$D" "$c" $dp 42 >> "$f"; done; done
         for dp in "ETTh2 96" "ETTm2 720"; do sf_ett "EPOCHS=2 SUITE=smoke-pn-ett-$D" 0.0005 $dp 42 >> "$f"; done
         sf_ett "EPOCHS=2 SEQ=336 SUITE=smoke-pn-ett-$D" 0.0005 ETTh2 96 42 >> "$f"
         g_ett "EPOCHS=2 SEQ=336 SUITE=smoke-pn-ett-$D" "COND=spkmx_pop_lif_muxlin_X NSCOPE=mixer" ETTh2 96 42 >> "$f"
         for c in "${GRID[@]}"; do g_mit "EPOCHS=1 SUITE=smoke-pn-mitbih-$D" "$c" 42 >> "$f"; done
         sf_mit "EPOCHS=1 SUITE=smoke-pn-mitbih-$D" 0.01 42 >> "$f"
         sf_mit "EPOCHS=1 VAL_FOLD=0 SUITE=smoke-pn-mitbih-$D" 0.01 42 >> "$f" ;;
  sf)    f="$Q/pn-sf-$D.txt"; : > "$f"
         for dp in "ETTm2 96" "ETTm2 720" "ETTh2 720" "ETTh2 96"; do
           for lr in 0.0001 0.00025 0.0005 0.001 0.002; do for s in 42 7 2026; do sf_ett "SUITE=flif-v1-ett-sfde" $lr $dp $s >> "$f"; done; done
         done
         for lr in 0.0025 0.005 0.01 0.02; do for s in 42 7 2026; do sf_mit "VAL_FOLD=0 SUITE=flif-v1-mitbih-sfde" $lr $s >> "$f"; done; done ;;
  main)  f="$Q/pn-main-$D.txt"; : > "$f"; lm=$(sel mitbih MITBIH) || exit 1
         for s in 42 7 2026; do sf_mit "SUITE=flif-v1-mitbih-sfde" "$lm" $s >> "$f"; done
         for dp in "ETTm2 96" "ETTm2 720" "ETTh2 720" "ETTh2 96"; do
           for c in "${GRID[@]}"; do for s in 42 7 2026; do g_ett "SUITE=flif-v1-ett-spkmux-pn" "$c" $dp $s >> "$f"; done; done
         done
         for c in "${GRID[@]}"; do for s in 42 7 2026; do g_mit "SUITE=flif-v1-mitbih-spkmux-pn" "$c" $s >> "$f"; done; done ;;
  len336) f="$Q/pn-len336-$D.txt"; : > "$f"
         for dp in "ETTm2 96" "ETTm2 720" "ETTh2 720" "ETTh2 96"; do
           set -- $dp; l=$(sel ett $1) || exit 1
           for s in 42 7 2026; do g_ett "SEQ=336 SUITE=flif-v1-ett-spkmux-pn336" "COND=spkmx_pop_lif_muxlin_X NSCOPE=mixer" $dp $s >> "$f"; done
           for s in 42 7 2026; do sf_ett "SEQ=336 SUITE=flif-v1-ett-spkmux-pn336" "$l" $dp $s >> "$f"; done
         done ;;
  sf336) f="$Q/pn-sf336-$D.txt"; : > "$f"
         for dp in "ETTm2 96" "ETTm2 720" "ETTh2 720" "ETTh2 96"; do
           set -- $dp; l=$(sel ett $1) || exit 1
           for s in 42 7 2026; do sf_ett "SEQ=336 SUITE=flif-v1-ett-spkmux-pn336" "$l" $dp $s >> "$f"; done
         done ;;
  *) echo "bad stage $1"; exit 1 ;;
esac
echo "$f:$(wc -l < "$f")"
