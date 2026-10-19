#!/usr/bin/env bash
# Writes scripts/queues/<SUITE>.txt (queue files are git-ignored; this script is the record). Stage 9 (MODI 2026-10-19).
# usage: make_queue_rhythm.sh <SUITE> <DATA> <kind> [LR]
#   kind main   : DATA=CINC2017 -> 10 conditions x seeds 42/7/2026 (fold 0); DATA=SHD -> 8 conditions x seeds 42/7/2026/123/1024
#   kind smoke  : every condition once, seed 42, EPOCHS=1
#   kind lr     : nsmt_lif, seed 42, LR in {0.001, 0.003}
#   kind crop   : memory pre-check, mix_lif_bidir and mix_none with CROP in {8, 32, 0}, seed 42
# The heaviest jobs (NSMT, ~35 GB) are listed first; the pool runs them one per GPU (see rhythm_driver.sh).
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; SUITE=${1:?}; DATA=${2:?}; KIND=${3:?}; LR=${4:-0.001}
mkdir -p "$W/scripts/queues"; Q="$W/scripts/queues/$SUITE.txt"; : > "$Q"
CONDS="nsmt_flif nsmt_pop nsmt_lif"; [ "$DATA" = CINC2017 ] && CONDS="$CONDS nsmt_lif_ce"
CONDS="$CONDS mix_pop_bidir mix_flif_bidir mix_lif_bidir"; [ "$DATA" = CINC2017 ] && CONDS="$CONDS mix_lif_bidir_ce"
CONDS="$CONDS spk_ssa mix_none"
SEEDS="42 7 2026"; [ "$DATA" = SHD ] && SEEDS="42 7 2026 123 1024"
C="DATA=$DATA SUITE=$SUITE LR=$LR"
case $KIND in
  main)  for c in $CONDS; do for s in $SEEDS; do echo "COND=$c $C bash nsmt_flif_v1/classification/scripts/run_rhythm.sh $s" >> "$Q"; done; done ;;
  smoke) for c in $CONDS; do echo "COND=$c $C EPOCHS=1 bash nsmt_flif_v1/classification/scripts/run_rhythm.sh 42" >> "$Q"; done ;;
  lr)    for lr in 0.001 0.003; do echo "COND=nsmt_lif DATA=$DATA SUITE=$SUITE LR=$lr bash nsmt_flif_v1/classification/scripts/run_rhythm.sh 42" >> "$Q"; done ;;
  crop)  for c in mix_lif_bidir mix_none; do for k in 8 32 0; do echo "COND=$c $C CROP=$k bash nsmt_flif_v1/classification/scripts/run_rhythm.sh 42" >> "$Q"; done; done ;;
  *) echo "bad kind $KIND"; exit 1 ;;
esac
echo "$Q: $(wc -l < "$Q") lines"
