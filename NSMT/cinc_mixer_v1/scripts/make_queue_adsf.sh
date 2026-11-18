#!/usr/bin/env bash
# make_queue_adsf.sh (2026-11-18, exp/f-lif-pop-v3; PROJECT_LOG AD SF-de pre-registration; user decision 11-18 "이대로 AD를 진행"):
# SF-de (the Spikformer-style model with the spikeDE LIF in every layer, NALL=lif) on anomaly detection, tuned exactly as the
# spikingjelly Spikformer-style model was on 11-02 / 11-03 (make_queue_ad.sh adtune_lr / adfinal): lr only, 2 mixer layers (L0),
# holdout tuning (first 80 % trained, last 20 % validated), then the whole training file at the selected lr. Suite flif-v1-ad-sfde.
#   repro   the default path of the edited runner: Spikformer-style (spikingjelly) MSL seed 42 at its tuned lr 2.5e-4, whole file   (1)
#   smoke   1 epoch, seed 42: SF-de x {MSL, SMAP, PSM} x {holdout, whole file}                                                      (6)
#   lr      holdout: lr {1e-4, 2.5e-4, 5e-4*, 1e-3, 2e-3} x 3 datasets x seeds 42 / 7 / 2026                                       (45)
#   final   whole file at the selected lr (results/flif-v1-ad-sfde/selected_lr.json, summarize_adsf.py --stage select)          (9)
# usage: bash make_queue_adsf.sh repro|smoke|lr|final ; prints <queue file>:<jobs>
set -eu
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="bash cinc_mixer_v1/scripts/run_ad_spk.sh"                 # args: <DS> <seed>, the pool adds the gpu
Q="$W/scripts/queues"; mkdir -p "$Q"; D=20261118
SUITE=flif-v1-ad-sfde
stage=${1:?repro|smoke|lr|final}
f="$Q/adsf-$stage-$D.txt"; : > "$f"
case $stage in
  repro) echo "HOLDOUT=0 LR=0.00025 SUITE=check-adsf-repro-$D COND=spk_ssa $RUN MSL 42" >> "$f" ;;
  smoke) for ds in MSL SMAP PSM; do for ho in 1 0; do
           echo "EPOCHS=1 HOLDOUT=$ho SUITE=smoke-adsf-$D COND=spk_ssa NALL=lif $RUN $ds 42" >> "$f"; done; done ;;
  lr)    for ds in SMAP PSM MSL; do for s in 42 7 2026; do for lr in 0.0001 0.00025 0.0005 0.001 0.002; do
           echo "HOLDOUT=1 LR=$lr SUITE=$SUITE COND=spk_ssa NALL=lif $RUN $ds $s" >> "$f"; done; done; done ;;
  final) sf="$W/results/$SUITE/selected_lr.json"
         [ -s "$sf" ] || { echo "no selection $sf" >&2; exit 1; }
         for ds in SMAP PSM MSL; do
           lr=$(python3 -c "import json; print(json.load(open('$sf'))['selected']['$ds'])") || exit 1
           for s in 42 7 2026; do echo "HOLDOUT=0 LR=$lr SUITE=$SUITE COND=spk_ssa NALL=lif $RUN $ds $s" >> "$f"; done
         done ;;
  *) echo "bad stage $stage"; exit 1 ;;
esac
echo "$f:$(wc -l < "$f")"
