#!/usr/bin/env bash
# MODI 2026-11-18: regenerates the (git-ignored) queue of the SpikeDE-basis comparison (docs/MODI_PROJECT_LOG.md; user decision 11-18
# "(가)": Spikformer-style and the base model both built from SpikeDE neurons only). Every new run uses the runner of its reference,
# so the only difference to the reference is --neuron_all lif.
#   suite flif-v1-ett-sde (36 runs): spkall_lif (run_ett_mixer.sh; reference spk_ssa, flif-v1-gen-ett) and
#         nsmt4q_pop_<lif|flif>_msn_sde (run_ett_dual.sh; reference nsmt4q_pop_<lif|flif>_msn, flif-v1-ett-ms);
#         ETTh2 / ETTm2 x H96 / H720, seeds 42 / 7 / 2026, input 96, every other setting unchanged
#   suite flif-v1-mitbih-sde (9 runs): spkall_lif (run_mitbih2.sh; reference spk_ssa, flif-v1-gen-mitbih) and
#         nsmt4q_pop_<lif|flif>_msn_sde (run_mitbih4.sh, lr 0.01, full training file; reference flif-v1-mitbih-ms); seeds 42 / 7 / 2026
#   suite flif-v1-mitbih-sde-repro (1 run, report only): spk_ssa seed 42 retrained with run_mitbih2.sh at this sha
# Order: the retrain run, ETTm2 (longest), MIT-BIH, ETTh2.
# NSMT/: bash scripts/gpu_pool2.sh <queue> <gpus> (appends the gpu).
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
SEEDS="42 7 2026"; ET=nsmt_flif_v1/forecasting/scripts/run_ett_dual.sh; MX=nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh
M2=nsmt_flif_v1/classification/scripts/run_mitbih2.sh; M4=nsmt_flif_v1/classification/scripts/run_mitbih4.sh
Q=$QD/sde-20261118.txt; : > $Q
echo "COND=spk_ssa SUITE=flif-v1-mitbih-sde-repro bash $M2 42" >> $Q
ett() { for pl in 720 96; do for s in $SEEDS; do
  echo "COND=spkall_lif SUITE=flif-v1-ett-sde bash $MX $1 $pl $s"
  for n in pop_lif pop_flif; do echo "COND=nsmt4q_${n}_msn_sde SUITE=flif-v1-ett-sde bash $ET $1 $pl $s"; done
done; done >> $Q; }
ett ETTm2
for s in $SEEDS; do
  echo "COND=spkall_lif SUITE=flif-v1-mitbih-sde bash $M2 $s"
  for n in pop_lif pop_flif; do echo "COND=nsmt4q_${n}_msn_sde SUITE=flif-v1-mitbih-sde bash $M4 $s"; done
done >> $Q
ett ETTh2
echo "$Q: $(grep -c . $Q) jobs"
