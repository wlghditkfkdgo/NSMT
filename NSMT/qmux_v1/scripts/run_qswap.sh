#!/usr/bin/env bash
# run_qswap.sh (2026-11-07, exp/f-lif-pop-v3; PROJECT_LOG query-swap pre-registration): one checkpoint through query_swap.py on one GPU. Queue line without the gpu argument, run from NSMT/.
# usage: [OUT=<results subfolder>] [EXTRA="--max_batches 2"] bash qmux_v1/scripts/run_qswap.sh <ett|mitbih> <raw.txt> <run_dir> <name> <gpu>
set -u
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
task=$1; raw=$2; run=$3; name=$4; gpu=$5
OUT=${OUT:-qswap/runs}
L="$W/log/qswap"; mkdir -p "$L"
cd "$W" && CUDA_VISIBLE_DEVICES=$gpu $PY -B -u query_swap.py --task "$task" --raw "$raw" --run_dir "$run" --name "$name" --out_dir "$W/results/$OUT" ${EXTRA:-} > "$L/$name.stdout" 2>&1
rc=$?
tail -1 "$L/$name.stdout" | cut -c1-240
echo "qswap rc=$rc $name"
exit $rc
