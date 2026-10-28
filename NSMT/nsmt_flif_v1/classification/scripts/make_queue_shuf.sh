#!/usr/bin/env bash
# MODI 2026-10-28: regenerates the (git-ignored) queue of the shuffle-retrain suite exactly as it is run: Spikformer-style and the
# four bidirectional OR Neo-mixers (LIF, f-LIF, population LIF, population f-LIF) trained and tested with shuffled token order
# (--token_shuffle via the _shuf suffix of run_rhythm_shuf.sh), CinC fold 0, 90 epochs, seeds 42 / 7 / 2026, long runs first.
# Pool (from NSMT/): bash scripts/gpu_pool2.sh nsmt_flif_v1/classification/scripts/queues/rhythm-shuf-20261028.txt 0,1,2,3,0,1,2,3
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; Q=$W/scripts/queues; mkdir -p "$Q"
: > $Q/rhythm-shuf-20261028.txt
for c in mix_pop_bidir mix_pop_lif_bidir mix_flif_bidir mix_lif_bidir spk_ssa; do for s in 42 7 2026; do
  echo "COND=${c}_shuf DATA=CINC2017 SUITE=flif-v1-rhythm-shuf-cinc LR=0.001 EPOCHS=90 bash nsmt_flif_v1/classification/scripts/run_rhythm_shuf.sh $s"
done; done >> $Q/rhythm-shuf-20261028.txt
wc -l < $Q/rhythm-shuf-20261028.txt
