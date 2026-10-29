#!/usr/bin/env bash
# MODI 2026-10-29: regenerates the (git-ignored) queues of the whole-model neuron swap batch (docs/MODI_PROJECT_LOG.md 2026-10-29).
#   smoke: every new condition once, 1 epoch, seed 42 (ETT ETTh2 H96, MIT-BIH, CinC fold 0)
#   quick: ETTh2 H96 -- control mixlb_<flif|pop_lif|pop_flif>_bidir and Spikformer-style spkall_<n>, seeds 42 / 7 / 2026 (21 runs)
#   main : CinC fold 0, 90 epochs -- B form spkneoall_<n> and its memoryless control spkneoall_<n>_nomem, Spikformer-style spkall_<n>;
#          MIT-BIH -- mixall_<n>_bidir, mixlb_<...>, spkall_<n>; ETT ETTh2 / ETTm2 x H96 / H720 (corrected 10-29: user decision 12 of
#          10-14; first written with 16 cells) -- mixall_<n>_bidir (ETTh2 H96 done in flif-v1-ett-nall)
#          and spkall_<n> (ETTh2 H96 in quick); seeds 42 / 7 / 2026; long CinC runs first, ETT H96 cells before longer horizons
# Pools (from NSMT/): scripts/nall_driver.sh
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; QD=$W/scripts/queues; mkdir -p $QD
NS="lif flif pop_lif pop_flif"; SEEDS="42 7 2026"
ETT=nsmt_flif_v1/forecasting/scripts/run_ett_mixer.sh; MIT=nsmt_flif_v1/classification/scripts/run_mitbih2.sh
CIN=nsmt_flif_v1/classification/scripts/run_rhythm_addneo.sh
MITC="$(for n in $NS; do echo -n "mixall_${n}_bidir "; done)mixlb_flif_bidir mixlb_pop_lif_bidir mixlb_pop_flif_bidir $(for n in $NS; do echo -n "spkall_$n "; done)"
Q=$QD/nall-smoke-20261029.txt; : > $Q
for c in mixlb_flif_bidir mixlb_pop_lif_bidir mixlb_pop_flif_bidir $(for n in $NS; do echo spkall_$n; done); do
  echo "COND=$c SUITE=smoke-nall-ett-20261029 EPOCHS=1 bash $ETT ETTh2 96 42"; done >> $Q
for c in $MITC; do echo "COND=$c SUITE=smoke-nall-mitbih-20261029 EPOCHS=1 bash $MIT 42"; done >> $Q
for n in $NS; do for c in spkneoall_$n spkneoall_${n}_nomem spkall_$n; do
  echo "COND=$c DATA=CINC2017 SUITE=smoke-nall-cinc-20261029 LR=0.001 EPOCHS=1 bash $CIN 42"; done; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
Q=$QD/nall-quick-20261029.txt; : > $Q
for c in mixlb_pop_flif_bidir mixlb_pop_lif_bidir mixlb_flif_bidir spkall_pop_flif spkall_pop_lif spkall_flif spkall_lif; do for s in $SEEDS; do
  echo "COND=$c SUITE=flif-v1-ett-nall bash $ETT ETTh2 96 $s"; done; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
Q=$QD/nall-20261029.txt; : > $Q
for n in pop_flif pop_lif flif lif; do for s in $SEEDS; do
  echo "COND=spkneoall_$n DATA=CINC2017 SUITE=flif-v1-rhythm-nall-cinc LR=0.001 EPOCHS=90 bash $CIN $s"; done; done >> $Q
for c in $MITC; do for s in $SEEDS; do echo "COND=$c SUITE=flif-v1-mitbih-nall bash $MIT $s"; done; done >> $Q
for n in pop_flif pop_lif flif lif; do for c in spkneoall_${n}_nomem spkall_$n; do for s in $SEEDS; do
  echo "COND=$c DATA=CINC2017 SUITE=flif-v1-rhythm-nall-cinc LR=0.001 EPOCHS=90 bash $CIN $s"; done; done; done >> $Q
for pl in 96 720; do for ds in ETTm2 ETTh2; do                     # user decision 12 (10-14): ETTh2 + ETTm2, H96 / H720
  for n in pop_flif pop_lif flif lif; do
    for c in mixall_${n}_bidir spkall_$n; do
      [ "$ds$pl" = ETTh296 ] && continue                       # done: mixall in the 11:01 quick test, spkall in the quick queue
      for s in $SEEDS; do echo "COND=$c SUITE=flif-v1-ett-nall bash $ETT $ds $pl $s"; done
    done
  done
done; done >> $Q
echo "$Q: $(grep -c . $Q) jobs"
