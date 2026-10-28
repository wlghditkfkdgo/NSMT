#!/usr/bin/env bash
# MODI 2026-10-28: regenerates the (git-ignored) queue of the PPG-DaLiA suite (42 runs, seeds 42 / 7 / 2026) exactly as it is run.
# Order: the first eight lines start together on workers 0,1,2,3,0,1,2,3 -> one B-form population run (about 19 GB) and one
# Neo-mixer population run per GPU; then the remaining heavy runs, then the light ones.
# Pool (from NSMT/): PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash scripts/gpu_pool2.sh \
#   nsmt_flif_v1/classification/scripts/queues/ppg-20261028.txt 0,1,2,3,0,1,2,3
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; Q=$W/scripts/queues/ppg-20261028.txt; mkdir -p "$(dirname "$Q")"
line() { echo "COND=$1 SUITE=flif-v1-ppg EPOCHS=100 bash nsmt_flif_v1/classification/scripts/run_ppg.sh $2"; }
{
  line ppg_spkneo_pop_flif 42; line ppg_spkneo_pop_lif 42; line ppg_spkneo_pop_flif 7; line ppg_spkneo_pop_lif 7
  line ppg_mix_pop_flif 42; line ppg_mix_pop_lif 42; line ppg_mix_pop_flif 7; line ppg_mix_pop_lif 7
  line ppg_spkneo_pop_flif 2026; line ppg_spkneo_pop_lif 2026; line ppg_mix_pop_flif 2026; line ppg_mix_pop_lif 2026
  for c in ppg_spkneo_flif ppg_mix_flif ppg_spkneo_lif ppg_spkneo_lif_nomem ppg_mix_lif ppg_ssa_full ppg_ssa_w32 ppg_ssa_w8 ppg_tok ppg_gru; do
    for s in 42 7 2026; do line $c $s; done
  done
} > "$Q"
wc -l < "$Q"
