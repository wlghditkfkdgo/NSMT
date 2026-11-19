#!/usr/bin/env bash
# sample_mem.sh <pid> <out>: per-process GPU memory (nvidia-smi, 1 s) until the process ends (2026-11-19, check-mxett)
P=$1; OUT=$2; : > "$OUT"
while kill -0 "$P" 2>/dev/null; do
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -v p="$P" -F', ' '$1==p {print strftime("%T"), $2}' >> "$OUT"
  sleep 1
done
