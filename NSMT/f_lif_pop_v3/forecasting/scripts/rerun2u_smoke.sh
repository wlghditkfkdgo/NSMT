#!/usr/bin/env bash
# Prereg 2U smoke test (NOT an experiment, not reported): the 2U pipeline end to end on 1-epoch runs so
# check_2u.py U9 has runs to check. Calibrations to their own folder, runs to their own suite, validation only.
#   1. smoke calibrations, ETTh1 R on, q1 and pearson         rerun_calibrate.sh smoke2u
#   2. fix them                                               rerun_common.py freeze-calibration
#   3. ETTh1 q1_R and pearson_R, seed 7, 1 epoch              rerun2u_ett.sh
# usage: rerun2u_smoke.sh <date tag> <gpu>
set -u
TAG=${1:?date tag}; GPU=${2:?gpu}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SUITE=rerun2u-smoke-$TAG; CALDIR=calibration-rerun2u-smoke-$TAG
[ -f "$W/results/$SUITE/calibration_raw.txt" ] || CALDIR=$CALDIR bash "$W/scripts/rerun_calibrate.sh" smoke2u "$SUITE" "$GPU"
[ -f "$W/results/$SUITE/calibration_manifest.json" ] || (cd "$W" && $PY rerun_common.py freeze-calibration "$SUITE" "$CALDIR")
for C in q1_R pearson_R; do
  SUITE=$SUITE EPOCHS=1 EXTRA="--calibration_dir $CALDIR" bash "$W/scripts/rerun2u_ett.sh" ETTh1 96 $C 7 "$GPU"
done
cat "$W/results/$SUITE/raw.txt"
