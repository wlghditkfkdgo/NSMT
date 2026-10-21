#!/usr/bin/env bash
# Stage 9 (MODI_PROJECT_LOG 2026-10-20): token-axis reach (rhythm_analysis.py) for every finished main run of the
# models that have a token-mixing network on the aligned axis (NSMT neocortex, Neo-mixer); report only.
# usage: run_rhythm_analysis.sh [GPUS=0,1,2,3] [PER_GPU=2]   -> results/<suite>/reach/<cond>_seed<s>/{reach_lag.csv,meta.json}; stdout in log/<suite>/reach/
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
IFS=, read -ra G <<< "${1:-0,1,2,3}"; PER=${2:-2}
CONDS=${CONDS:-'nsmt_lif nsmt_flif nsmt_pop mix_lif_bidir mix_flif_bidir mix_pop_bidir'}   # MODI 2026-10-21: overridable, also SUITES
JOBS=$(mktemp)
for SUITE in ${SUITES:-flif-v1-rhythm-cinc flif-v1-rhythm-shd}; do
  for C in $CONDS; do
    grep '^RESULT' "$W/results/$SUITE/raw.txt" | grep " cond=$C " | grep ' crop=0 ' | while read -r L; do
      S=$(grep -oE ' seed=[0-9]+' <<< "$L" | cut -d= -f2); D=$(grep -oE ' dir=\S+' <<< "$L" | cut -d= -f2)
      echo "$SUITE $C $S $D"
    done
  done
done > "$JOBS"
echo "$(wc -l < "$JOBS") runs"
i=0
while read -r SUITE C S D; do
  GPU=${G[$(( i % ${#G[@]} ))]}; i=$((i + 1))
  mkdir -p "$W/results/$SUITE/reach" "$W/log/$SUITE/reach"
  echo "$GPU $SUITE $C $S $D"
done < "$JOBS" | xargs -P $(( ${#G[@]} * PER )) -L 1 bash -c '
  cd "'"$W"'" && CUDA_VISIBLE_DEVICES=$0 OMP_NUM_THREADS=4 '"$PY"' ./rhythm_analysis.py --run_dir "$4" --name "$2_seed$3" \
    --out_dir "'"$W"'/results/$1/reach" > "'"$W"'/log/$1/reach/$2_seed$3.stdout" 2>&1 \
    && echo "done $1 $2 seed$3" || echo "FAILED $1 $2 seed$3"'
rm -f "$JOBS"
