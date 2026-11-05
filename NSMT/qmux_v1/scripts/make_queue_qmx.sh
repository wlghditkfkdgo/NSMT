#!/usr/bin/env bash
# make_queue_qmx.sh (2026-11-05, exp/f-lif-pop-v3; PROJECT_LOG bidirectional Neocortex pre-registration): queue files of the query form with
# the bidirectional Neocortex (neo_mux.py). gpu_pool2.sh lines without the gpu argument, run from NSMT/.
#   qmx_repro  this folder's code reruns source runs: nsmt4q_lif ETTh2 H96 seed 42, nsmt4q_pop_lif MIT-BIH seed 42 (check_qmx.py --after_repro)
#   qmx_smoke  1 epoch, seed 42: 16 conditions on ETTh2 H96 and MIT-BIH, the 4 muxlin-OR-residual ones on ETTm2 H720
#   qmx_ett    4 neurons x {muxlin, or} x {or, iand} x ETTh2 / ETTm2 x H96 / H720 x seeds 42 / 7 / 2026 = 192, + the query controls the
#              source did not run: nsmt4q_<lif|flif> on ETTm2 x H96 / H720 x seeds = 12; LIF first, f-LIF neurons last
#   qmx_mitbih 4 neurons x {muxlin, or} x {or, iand} x seeds = 48, + nsmt4q_<lif|flif> x seeds = 6
# usage: bash make_queue_qmx.sh qmx_repro|qmx_smoke|qmx_ett|qmx_mitbih
set -eu
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
Q="$W/scripts/queues"; mkdir -p "$Q"; D=$(date +%Y%m%d)
stage=${1:?qmx_repro|qmx_smoke|qmx_ett|qmx_mitbih}
f="$Q/$stage-$D.txt"; : > "$f"
E="bash qmux_v1/scripts/run_ett_qmx.sh"; M="bash qmux_v1/scripts/run_mitbih_qmx.sh"
NEURONS="lif pop_lif flif pop_flif"; SEEDS="42 7 2026"
case $stage in
  qmx_repro)
    echo "SUITE=check-qmx-repro-$D COND=nsmt4q_lif $E ETTh2 96 42" >> "$f"
    echo "SUITE=check-qmx-repro-$D COND=nsmt4q_pop_lif $M 42" >> "$f" ;;
  qmx_smoke)
    for n in $NEURONS; do for fu in muxlin or; do for r in or iand; do
      echo "EPOCHS=1 SUITE=smoke-qmx-$D COND=qmx_${n}_${fu}_${r} $E ETTh2 96 42" >> "$f"
      echo "EPOCHS=1 SUITE=smoke-qmx-$D COND=qmx_${n}_${fu}_${r} $M 42" >> "$f"
    done; done; echo "EPOCHS=1 SUITE=smoke-qmx-$D COND=qmx_${n}_muxlin_or $E ETTm2 720 42" >> "$f"; done ;;
  qmx_ett)
    for n in $NEURONS; do for ds in ETTh2 ETTm2; do for pl in 96 720; do for s in $SEEDS; do
      for fu in muxlin or; do for r in or iand; do
        echo "SUITE=qmx-ett COND=qmx_${n}_${fu}_${r} $E $ds $pl $s" >> "$f"
      done; done
      case $n$ds in lifETTm2|flifETTm2) echo "SUITE=qmx-ett COND=nsmt4q_$n $E $ds $pl $s" >> "$f" ;; esac
    done; done; done; done ;;
  qmx_mitbih)
    for n in $NEURONS; do for s in $SEEDS; do
      for fu in muxlin or; do for r in or iand; do
        echo "SUITE=qmx-mitbih COND=qmx_${n}_${fu}_${r} $M $s" >> "$f"
      done; done
      case $n in lif|flif) echo "SUITE=qmx-mitbih COND=nsmt4q_$n $M $s" >> "$f" ;; esac
    done; done ;;
  *) echo "bad stage $stage"; exit 1 ;;
esac
echo "$f: $(grep -c . "$f") jobs"
