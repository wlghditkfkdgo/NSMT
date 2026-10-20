#!/usr/bin/env bash
# Prereg 2T smoke test (NOT an experiment; nothing here is reported as a result): the rerun pipeline end
# to end on 1-epoch runs, so analysis/rerun_gate_check.py has new-neuron runs to check and to inject faults
# into. Calibrations go to their own folder; runs to their own suite. Reads no split beyond validation.
#   1. smoke calibrations (5 recall on CPU, 2 ETTh1 with R on GPU)      rerun_calibrate.sh smoke
#   2. fix them (name + SHA256)                                         rerun_common.py freeze-calibration
#   3. ETTh1 q1_R and pearson_R, seed 7, 1 epoch                        rerun_ett.sh
#   4. recall selection stage, seed 7 x q {1, 0.1, 0.25, 0.5}, 1 epoch  rerun_recall.sh select
# usage: rerun_smoke.sh <date tag> <gpu>        e.g. rerun_smoke.sh 20261021 0
set -u
TAG=${1:?date tag}; GPU=${2:?gpu}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
SUITE=rerun-smoke-$TAG; CALDIR=calibration-rerun-smoke-$TAG
[ -f "$W/results/$SUITE/calibration_raw.txt" ] || CALDIR=$CALDIR bash "$W/scripts/rerun_calibrate.sh" smoke "$SUITE" "$GPU"
[ -f "$W/results/$SUITE/calibration_manifest.json" ] || (cd "$W" && $PY rerun_common.py freeze-calibration "$SUITE" "$CALDIR")
for C in q1_R pearson_R; do
  SUITE=$SUITE EPOCHS=1 EXTRA="--calibration_dir $CALDIR" bash "$W/scripts/rerun_ett.sh" ETTh1 96 $C 7 "$GPU"
done
EPOCHS=1 EXTRA="--calibration_dir $CALDIR" bash "$W/scripts/rerun_recall.sh" select "$SUITE"
cat "$W/results/$SUITE/raw.txt"
