#!/usr/bin/env bash
# MODI 2026-11-26: one run of the order / accumulation check (order_check.py; docs/MODI_PROJECT_LOG.md 2026-11-26).
# usage (from NSMT/, the pool appends the gpu):  SUITE=<suite> bash nsmt_flif_v1/classification/scripts/run_oc.sh <DS> <cond> <seed|fold> <gpu>
#   DS = SC | MITBIH | SLEEPIN | SLEEPCTX; for SC / MITBIH the third argument is the seed, for SLEEP* it is the fold (seed 42).
# env: SUITE (default flif-v1-oc), EPOCHS (default 30), MAXB (smoke only: batches per epoch), PY.
# Writes one RESULT (or SKIP) line to nsmt_flif_v1/classification/results/<suite>/raw.txt.
set -u
DS=$1; COND=$2; A=$3; GPU=$4
PY=${PY:-/home/yschoi/.conda/envs/snn_recall/bin/python}
export LD_LIBRARY_PATH="$(dirname "$(dirname "$PY")")/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
C=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SUITE=${SUITE:-flif-v1-oc}; EPOCHS=${EPOCHS:-30}; MAXB=${MAXB:-0}
case $DS in SC|MITBIH) SEED=$A; FOLD=0; ID=s$A ;; SLEEPIN|SLEEPCTX) SEED=42; FOLD=$A; ID=f$A ;; *) echo "bad DS $DS"; exit 2 ;; esac
SHA=$(cd "$C" && git rev-parse --short HEAD); TS=$(date +%Y%m%d)
RES=$C/results/$SUITE; mkdir -p "$RES" "$C/log/$SUITE/stdout"
D=$C/log/$SUITE/$DS/$TS/${COND}+seed${SEED}+fold${FOLD}+ep${EPOCHS}+$SHA
OUT=$C/log/$SUITE/stdout/${DS}_${COND}_${ID}_ep${EPOCHS}_${TS}_${SHA}.stdout
[ -e "$D" ] && { echo "SKIP(run folder exists) ds=$DS cond=$COND seed=$SEED fold=$FOLD dir=$D" | tee -a "$RES/raw.txt"; exit 1; }
t0=$(date +%s)
(cd "$C" && CUDA_VISIBLE_DEVICES=$GPU "$PY" order_check.py --ds "$DS" --cond "$COND" --seed "$SEED" --fold "$FOLD" --epochs "$EPOCHS" \
   --max_batches "$MAXB" --log_dir "$D" > "$OUT" 2>&1); RC=$?
t1=$(date +%s)
CSV=$D/log/final+result.csv
if [ $RC -ne 0 ] || [ ! -s "$CSV" ]; then echo "SKIP(rc=$RC, no result) ds=$DS cond=$COND seed=$SEED fold=$FOLD out=$OUT" | tee -a "$RES/raw.txt"; exit 1; fi
VALS=$("$PY" - "$CSV" <<'PYEOF'
import csv, sys
r = next(csv.DictReader(open(sys.argv[1])))
print(' '.join(f'{k}={v}' for k, v in r.items()))
PYEOF
)
echo "RESULT exp=order_check ds=$DS cond=$COND seed=$SEED fold=$FOLD epochs=$EPOCHS maxb=$MAXB $VALS wall_s=$((t1-t0)) sha=$SHA dir=$D out=$OUT" | tee -a "$RES/raw.txt"
