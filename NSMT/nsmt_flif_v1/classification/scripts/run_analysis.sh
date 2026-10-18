#!/usr/bin/env bash
# Runs cls_analysis.py on one trained MIT-BIH run (looked up in results/<suite>/raw.txt).
# usage: run_analysis.sh <suite> <cond> <seed> <gpu>
set -u
SUITE=${1:?}; COND=${2:?}; S=${3:?}; GPU=${4:?}
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
D=$(grep "^RESULT .* cond=$COND .*seed=$S " "$W/results/$SUITE/raw.txt" | tail -1 | grep -oE 'dir=\S+' | cut -d= -f2)
[ -z "$D" ] && { echo "ANALYSIS-SKIP no run for cond=$COND seed=$S"; exit 1; }
mkdir -p "$W/log/$SUITE/analysis"
(cd "$W" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=${THREADS:-4} $PY cls_analysis.py --run_dir "$D" --name ${COND}_seed$S \
   --out_dir "$W/results/$SUITE/analysis" > "$W/log/$SUITE/analysis/${COND}_seed$S.stdout" 2>&1) \
  && echo "ANALYSIS-DONE cond=$COND seed=$S" || { echo "ANALYSIS-FAIL cond=$COND seed=$S"; exit 1; }
