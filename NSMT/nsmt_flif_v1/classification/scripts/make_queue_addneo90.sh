#!/usr/bin/env bash
# MODI 2026-10-27: regenerates the (git-ignored) queue of B at 90 epochs (Spikformer-style + the T = N neocortex added as extra
# keys / values; the same five conditions as flif-v1-rhythm-addneo-cinc, only EPOCHS=90), long runs first, seeds 42 / 7 / 2026,
# CinC fold 0. Pool: from NSMT/,  bash scripts/gpu_pool2.sh <queue> 0,1,0,1   (2 per GPU on GPUs 0 and 1)
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; Q=$W/scripts/queues/rhythm-addneo90-20261027.txt
: > $Q
for c in spkneo_pop_flif spkneo_pop_lif spkneo_flif spkneo_lif spkneo_lif_nomem; do for s in 42 7 2026; do
  echo "COND=$c DATA=CINC2017 SUITE=flif-v1-rhythm-addneo90-cinc LR=0.001 EPOCHS=90 bash nsmt_flif_v1/classification/scripts/run_rhythm_addneo.sh $s"
done; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
