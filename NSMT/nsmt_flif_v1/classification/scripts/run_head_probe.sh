#!/usr/bin/env bash
# MODI 2026-11-03: head readout diagnostic (docs/MODI_PROJECT_LOG.md 2026-11-03) -- head_probe.py on every checkpoint of one group:
#   mitbih : nsmt4q_pop_{lif,flif}, lr 0.01, --mitbih_val runs of results/flif-v1-mitbih-query-tune/raw.txt (6)
#   etth2  : nsmt4q_pop_{lif,flif} ETTh2 H96 / H720 runs of forecasting/results/flif-v1-ett-query/raw.txt (12)
#   ettm2  : nsmt4q_pop_{lif,flif} ETTm2 H96 / H720 runs of forecasting/results/flif-v1-ett-query-m2/raw.txt (12)
# A checkpoint whose JSON exists is skipped; a failed check leaves no JSON (FAIL line, see the stdout file).
# usage: run_head_probe.sh <mitbih|etth2|ettm2> <gpu>   env: PY, OUT, CHECK=1 (--check_only, R0 only; default OUT results/checks/head-probe-20261103)
set -u
G=${1:?}; GPU=${2:?}
C="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # nsmt_flif_v1/classification
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
case $G in
  mitbih) RAW=$C/results/flif-v1-mitbih-query-tune/raw.txt; TASK=mitbih; SEL='lr=0.01 val=1 ' ;;
  etth2)  RAW=$C/../forecasting/results/flif-v1-ett-query/raw.txt; TASK=ett; SEL='ds=ETTh2 ' ;;
  ettm2)  RAW=$C/../forecasting/results/flif-v1-ett-query-m2/raw.txt; TASK=ett; SEL='ds=ETTm2 ' ;;
  *) echo "bad group $G"; exit 1 ;;
esac
RAW="$(cd "$(dirname "$RAW")" && pwd)/raw.txt"
if [ -n "${CHECK:-}" ]; then OUT=${OUT:-$C/results/checks/head-probe-20261103}; else OUT=${OUT:-$C/results/flif-v1-head-probe}; fi
LOG=$C/log/flif-v1-head-probe; mkdir -p "$OUT" "$LOG"
grep '^RESULT' "$RAW" | grep -E 'cond=nsmt4q_pop_(lif|flif) ' | grep -F "$SEL" | while read -r line; do
  cond=$(echo "$line" | grep -oP 'cond=\K\S+'); seed=$(echo "$line" | grep -oP ' seed=\K\S+'); dir=$(echo "$line" | grep -oP 'dir=\K\S+')
  if [ $TASK = mitbih ]; then name=${cond}_mitbih_seed$seed
  else name=${cond}_$(echo "$line" | grep -oP 'ds=\K\S+')_p$(echo "$line" | grep -oP ' pl=\K\S+')_seed$seed; fi
  [ -n "${CHECK:-}" ] && name=${name}_check
  [ -s "$OUT/$name.json" ] && { echo "have $name"; continue; }
  if (cd "$C" && CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=8 $PY ./head_probe.py --task $TASK --run_dir "$dir" --raw "$RAW" --name $name \
        --out_dir "$OUT" ${CHECK:+--check_only} < /dev/null > "$LOG/$name.stdout" 2>&1); then echo "ok $name"
  else echo "FAIL $name (see $LOG/$name.stdout)"; fi
done
