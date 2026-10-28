#!/bin/bash
# run_cause.sh (MODI 2026-10-28): cause analysis without training. rhythm_cause.py --kind b over the B runs at 90 epochs
# (flif-v1-rhythm-addneo90-cinc, 5 conditions x 3 seeds) and --kind h5 over Spikformer-style and the four Neo-mixers at 90 epochs
# (flif-v1-rhythm-budget-cinc, 5 x 3); one job per GPU at a time; then summarize_cause.py. Run directories come from the RESULT
# lines (dir=...) of the two suites' raw.txt. Console output goes to log/<suite>/<name>.log (local), one line per run to
# results/<suite>/pool_runs.txt. Honours PY, GPUS, SUITE.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
GPUS=${GPUS:-"0 1 2 3"}
SUITE=${SUITE:-flif-v1-cause-cinc}
export LD_LIBRARY_PATH=/home/yschoi/.conda/envs/snn_recall/lib
RES=$HERE/results/$SUITE; LOG=$HERE/log/$SUITE
mkdir -p "$RES" "$LOG"
LIST=$LOG/runs.txt; : > "$LIST"
add() {   # kind suite cond...
  local kind=$1 src=$2; shift 2
  for c in "$@"; do
    grep "^RESULT .* cond=$c " "$HERE/results/$src/raw.txt" | while read -r line; do
      s=$(echo "$line" | grep -o ' seed=[0-9]*' | cut -d= -f2); d=$(echo "$line" | grep -o ' dir=[^ ]*' | cut -d= -f2)
      echo "$kind ${c}_seed$s $d" >> "$LIST"
    done
  done
}
add b flif-v1-rhythm-addneo90-cinc spkneo_lif spkneo_lif_nomem spkneo_flif spkneo_pop_lif spkneo_pop_flif
add h5 flif-v1-rhythm-budget-cinc spk_ssa mix_lif_bidir mix_flif_bidir mix_pop_lif_bidir mix_pop_bidir
n=$(wc -l < "$LIST"); echo "$n runs"; [ "$n" -eq 30 ] || { echo "expected 30 runs"; exit 1; }
i=0; ng=$(echo $GPUS | wc -w)
for g in $GPUS; do
  i=$((i + 1))
  (awk -v k=$i -v n=$ng 'NR % n == k % n' "$LIST" | while read -r kind name dir; do
     t0=$(date +%s)
     if CUDA_VISIBLE_DEVICES=$g "$PY" "$HERE/rhythm_cause.py" --kind "$kind" --run_dir "$dir" --name "$name" --out_dir "$RES" \
          > "$LOG/$name.log" 2>&1; then st=OK; else st=FAIL; fi
     echo "$st $name gpu=$g start=$(date -d @$t0 '+%F %T') sec=$(( $(date +%s) - t0 ))" >> "$RES/pool_runs.txt"
   done) &
done
wait
"$PY" "$HERE/summarize_cause.py" --suite "$SUITE" > "$LOG/summary.log" 2>&1
echo "ALL DONE $(date '+%F %T')" >> "$RES/pool_runs.txt"
