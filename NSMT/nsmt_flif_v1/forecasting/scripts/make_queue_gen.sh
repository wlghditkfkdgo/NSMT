#!/usr/bin/env bash
# MODI 2026-10-27: regenerates the (git-ignored) queue of the generalization batch: the 9 CinC conditions on MIT-BIH (27 runs, run
# first) and on ETT (ETTh1 / ETTh2 / ETTm1 / ETTm2 x H 96 / 192 / 336 / 720, 432 runs), seeds 42 / 7 / 2026, heavier runs first.
# Pool: from NSMT/,  bash scripts/gpu_pool2.sh nsmt_flif_v1/forecasting/scripts/queues/gen-20261027.txt 2,3,2,3,2,3,2,3,2,3
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; Q=$W/scripts/queues/gen-20261027.txt
mkdir -p "$W/scripts/queues"; : > $Q
CONDS="mixlinmx_pop_flif mix_pop_flif_bidir mixlinmx_flif mix_flif_bidir mixlinmx_pop_lif mix_pop_lif_bidir mixlinmx_lif mix_lif_bidir spk_ssa"
for c in $CONDS; do for s in 42 7 2026; do
  echo "COND=$c SUITE=flif-v1-gen-mitbih bash nsmt_flif_v1/classification/scripts/run_mitbih2.sh $s"
done; done >> $Q
for ds in ETTm1 ETTm2 ETTh1 ETTh2; do for pl in 720 336 192 96; do for c in $CONDS; do for s in 42 7 2026; do
  echo "COND=$c SUITE=flif-v1-gen-ett bash nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh $ds $pl $s"
done; done; done; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
