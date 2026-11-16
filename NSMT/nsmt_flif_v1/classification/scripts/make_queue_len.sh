#!/usr/bin/env bash
# MODI 2026-11-16: regenerates the (git-ignored) queue of the input-length test (docs/MODI_PROJECT_LOG.md; user decision 11-16: "입력길이를
# 336으로 보고 나서 720을 확인"). Does the gap between the base model (Neocortex-MS query population, msn) and Spikformer-style grow
# when the input window (= number of tokens) grows?  ETTh2 / ETTm2 x H96 / H720, seeds 42 / 7 / 2026, every other setting unchanged.
#   usage: make_queue_len.sh <SEQ: 336 | 720>
#   suite flif-v1-ett-len<SEQ> (84 runs): spk_ssa (run_ett_mixer.sh), nsmt4q_pop_<lif|flif>_msn, .._msn_nomem (--neo_memoryless),
#         .._msn_nshuf (--neo_shuffle) (run_ett_dual.sh), all with SEQ=<SEQ>
#   SEQ 336 only: suite flif-v1-ett-len96ctl (48 runs): the two controls at the default input length 96 (their reference; the base
#         and Spikformer-style runs at 96 are flif-v1-ett-ms and flif-v1-gen-ett)
# Order: the main comparison first (Spikformer-style + base; ETTm2 then ETTh2), then the controls at <SEQ>, then the controls at 96.
# NSMT/: bash scripts/gpu_pool2.sh <queue> <gpus> (appends the gpu).
set -u
SEQ=${1:?SEQ 336 | 720}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"; ET=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh; MX=nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh
Q=$QD/len$SEQ-20261116.txt; : > $Q
SU=flif-v1-ett-len$SEQ
for ds in ETTm2 ETTh2; do for pl in 720 96; do for s in $SEEDS; do
  echo "COND=spk_ssa SEQ=$SEQ SUITE=$SU bash $MX $ds $pl $s"
  for n in pop_lif pop_flif; do echo "COND=nsmt4q_${n}_msn SEQ=$SEQ SUITE=$SU bash $ET $ds $pl $s"; done
done; done; done >> $Q
for ds in ETTm2 ETTh2; do for pl in 720 96; do for v in nomem nshuf; do for n in pop_lif pop_flif; do for s in $SEEDS; do
  echo "COND=nsmt4q_${n}_msn_${v} SEQ=$SEQ SUITE=$SU bash $ET $ds $pl $s"
done; done; done; done; done >> $Q
if [ "$SEQ" = 336 ]; then
  for ds in ETTm2 ETTh2; do for pl in 720 96; do for v in nomem nshuf; do for n in pop_lif pop_flif; do for s in $SEEDS; do
    echo "COND=nsmt4q_${n}_msn_${v} SUITE=flif-v1-ett-len96ctl bash $ET $ds $pl $s"
  done; done; done; done; done >> $Q
fi
echo "$Q: $(grep -c . $Q) jobs"
