#!/usr/bin/env bash
# make_queue_ms.sh (2026-11-08, exp/f-lif-pop-v3; PROJECT_LOG membrane shortcut pre-registration): queues of the membrane shortcut test, LIF only.
# Settings are the tuned ones, not re-tuned:
#   ETT      SpkMixer muxlin, 1 mixer layer, lr 5e-4 ETTh2 / 1e-4 ETTm2;  Spikformer-style lr 5e-4 ETTh2 / 2.5e-4 ETTm2
#   MIT-BIH  SpkMixer muxlin, 2 mixer layers, lr 0.02;                    Spikformer-style lr 0.005
#   repro : default paths again, one seed: must match the control rows in every shared field but time, sha, dir (check_ms.py --repro)
#   smoke : with the shortcut, 2 epochs ETT (ETTh2 H96, ETTm2 H720) and 1 epoch MIT-BIH, both models, seed 42
#   ett   : 4 cells x 3 seeds x 2 models = 24 runs (suite flif-v1-ett-spkmux-ms)
#   mitbih: 3 seeds x 2 models = 6 runs (suite flif-v1-mitbih-spkmux-ms)
# usage (from anywhere): bash make_queue_ms.sh repro|smoke|ett|mitbih ; prints <queue file>:<jobs>
set -u
Q="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/queues"; mkdir -p "$Q"
E="bash cinc_mixer_v1/scripts/run_ett_spk.sh"
M="bash cinc_mixer_v1/scripts/run_mitbih_spk.sh"
D=20261108
lr_mx() { case $1 in ETTh2) echo 0.0005 ;; ETTm2) echo 0.0001 ;; esac; }
lr_sf() { case $1 in ETTh2) echo 0.0005 ;; ETTm2) echo 0.00025 ;; esac; }
ly_sf() { case $1 in ETTh2) echo 1 ;; ETTm2) echo 2 ;; esac; }
case ${1:?repro|smoke|ett|mitbih} in
  repro) f="$Q/ms-repro-$D.txt"; : > "$f"; S=check-ms-repro-$D
         echo "SUITE=$S COND=spkmx_lif_muxlin_add LR=0.02 MLAYERS=2 $M 42" >> "$f"
         echo "SUITE=$S COND=spk_ssa LR=0.005 MLAYERS=2 $M 42" >> "$f"
         echo "SUITE=$S-ett COND=spkmx_lif_muxlin_iand LR=$(lr_mx ETTh2) LAYERS=1 $E ETTh2 96 42" >> "$f"
         echo "SUITE=$S-ett COND=spk_ssa LR=$(lr_sf ETTh2) LAYERS=$(ly_sf ETTh2) $E ETTh2 96 42" >> "$f" ;;
  smoke) f="$Q/ms-smoke-$D.txt"; : > "$f"
         for dp in "ETTh2 96" "ETTm2 720"; do ds=${dp% *}
           echo "EPOCHS=2 SUITE=smoke-ms-ett-$D COND=spkmx_lif_muxlin_ms LR=$(lr_mx $ds) LAYERS=1 $E $dp 42" >> "$f"
           echo "EPOCHS=2 SUITE=smoke-ms-ett-$D COND=spk_ssa MS=1 LR=$(lr_sf $ds) LAYERS=$(ly_sf $ds) $E $dp 42" >> "$f"
         done
         echo "EPOCHS=1 SUITE=smoke-ms-mitbih-$D COND=spkmx_lif_muxlin_ms LR=0.02 MLAYERS=2 $M 42" >> "$f"
         echo "EPOCHS=1 SUITE=smoke-ms-mitbih-$D COND=spk_ssa MS=1 LR=0.005 MLAYERS=2 $M 42" >> "$f" ;;
  ett)   f="$Q/ms-ett-$D.txt"; : > "$f"
         for dp in "ETTm2 720" "ETTm2 96" "ETTh2 720" "ETTh2 96"; do ds=${dp% *}
           for s in 42 7 2026; do
             echo "SUITE=flif-v1-ett-spkmux-ms COND=spkmx_lif_muxlin_ms LR=$(lr_mx $ds) LAYERS=1 $E $dp $s" >> "$f"
             echo "SUITE=flif-v1-ett-spkmux-ms COND=spk_ssa MS=1 LR=$(lr_sf $ds) LAYERS=$(ly_sf $ds) $E $dp $s" >> "$f"
           done
         done ;;
  mitbih) f="$Q/ms-mitbih-$D.txt"; : > "$f"
         for s in 42 7 2026; do
           echo "SUITE=flif-v1-mitbih-spkmux-ms COND=spkmx_lif_muxlin_ms LR=0.02 MLAYERS=2 $M $s" >> "$f"
           echo "SUITE=flif-v1-mitbih-spkmux-ms COND=spk_ssa MS=1 LR=0.005 MLAYERS=2 $M $s" >> "$f"
         done ;;
  *) echo "bad stage $1"; exit 1 ;;
esac
echo "$f:$(wc -l < "$f")"
